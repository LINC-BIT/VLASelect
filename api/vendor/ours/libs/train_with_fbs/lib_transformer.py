from typing import Optional
import torch
import tqdm
from torch import nn
from ours.utils.dl.common.model import get_model_device, get_model_latency, get_model_size, get_module, get_super_module, set_module
from ours.utils.common.log import logger
from einops.layers.torch import Rearrange


class KTakesAll(nn.Module):
    def __init__(self, k):
        super(KTakesAll, self).__init__()

        self.k = k
        
    def forward(self, g: torch.Tensor):
        k = int(g.size(1) * self.k)
        
        if k == 0: # fuck this corner case, which runs correctly under pytorch 1.7.0 but completely incorrectly under pytorch 1.10.0
            self.cached_i = torch.LongTensor([[]]).to(g.device)
            return g
        
        i = (-g).topk(k, 1)[1]
        self.cached_i = i
        t = g.scatter(1, i, 0)
        return t
    
    def __repr__(self):
        return f'KTakesAll(k={self.k})'
    

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
        
    def forward(self, x):
        return x.abs()
    
    
class SqueezeLast(nn.Module):
    def __init__(self):
        super(SqueezeLast, self).__init__()
    
    def forward(self, x):
        return x.squeeze(-1)
    
    
class SwinWindowMerge(nn.Module):
    def __init__(self, window_merge):
        super(SwinWindowMerge, self).__init__()
        self.window_merge = window_merge
    
    def forward(self, x):
        return x.view(x.size(0) // self.window_merge, self.window_merge, *list(x.size())[1:]).mean(1)
    
    
# class SwinWindowExpand(nn.Module):
#     def __init__(self, window_merge):
#         super(SwinWindowExpand, self).__init__()
#         self.window_merge = window_merge
    
#     def forward(self, x):
#         return x.repeat(self.window_merge, 1)
    
import os
    
class LinearWithFBS(nn.Module):
    def __init__(self, raw_linear: nn.Linear, k: float, r: int, window_merge=None):
        super(LinearWithFBS, self).__init__()
        
        self.window_merge = window_merge
        
        if window_merge is None:
            self.fbs = nn.Sequential(
                Rearrange('b n d -> b d n'),
                Abs(),
                nn.AdaptiveAvgPool1d(1),
                SqueezeLast(),
                nn.Linear(raw_linear.in_features, raw_linear.out_features // r),
                nn.ReLU(),
                nn.Linear(raw_linear.out_features // r, raw_linear.out_features),
                nn.ReLU(),
            )
            nn.init.constant_(self.fbs[6].bias, 1.)
            nn.init.kaiming_normal_(self.fbs[6].weight)
        else:
            self.fbs = nn.Sequential(
                SwinWindowMerge(window_merge),
                Rearrange('b n d -> b d n'),
                Abs(),
                nn.AdaptiveAvgPool1d(1),
                SqueezeLast(),
                nn.Linear(raw_linear.in_features, raw_linear.out_features // r),
                nn.ReLU(),
                nn.Linear(raw_linear.out_features // r, raw_linear.out_features),
                nn.ReLU()
            )
            nn.init.constant_(self.fbs[7].bias, 1.)
            nn.init.kaiming_normal_(self.fbs[7].weight)
            
        self.k_takes_all = KTakesAll(k)
        
        self.raw_linear = raw_linear
        
        self.cached_raw_w = None
        self.l1_reg_of_raw_w = None
        self.cached_w = None
        
    def forward(self, x):
        if x.dim() == 3:
            os.environ['_Z_BATCH_SIZE'] = str(x.size(0))
            resize = False
        elif x.dim() == 2:
            x = x.view(int(os.environ['_Z_BATCH_SIZE']), x.size(0) // int(os.environ['_Z_BATCH_SIZE']), x.size(1))
            resize = True
            
            
        raw_x = self.raw_linear(x)
        raw_w = self.fbs(x)
        
        self.cached_raw_w = raw_w
        self.l1_reg_of_raw_w = raw_w.norm(1)
        
        w = self.k_takes_all(raw_w)
        self.cached_w = w
        
        if hasattr(self, 'window_merge') and self.window_merge is not None:
            return raw_x * w.repeat(self.window_merge, 1).unsqueeze(1)
        
        res = raw_x * w.unsqueeze(1)
        if resize:
            res = res.view(-1, res.size(-1))
        return res

def add_FBS(model: nn.Module, init_k: float, r: int, ignore_layers=None, perf_test=True, example_input=None, window_merges=None):
    device = get_model_device(model)
    model.eval()

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (ignore_layers is not None and name not in ignore_layers):
            linear_with_fbs = LinearWithFBS(module, init_k, r, window_merges[name] if window_merges is not None else None).to(device)
            set_module(model, name, linear_with_fbs)
            
    logger.debug(model)
    
    return model


def get_l1_reg_in_model(model_with_fbs):
    res = 0.
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, LinearWithFBS):
            res += module.l1_reg_of_raw_w
    return res


def set_sparsity(model, k):
    for name, module in model.named_modules():
        if isinstance(module, KTakesAll):
            module.k = k


def clear_cache(model_with_fbs):
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, LinearWithFBS):
            module.cached_raw_w = None
            module.cached_w = None
            module.l1_reg_of_raw_w = None
            
            
def get_fbs_params(model_with_fbs):
    res = []
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, LinearWithFBS):
            res += list(module.fbs.parameters())
    return res

def get_fbs_module_names(model_with_fbs):
    res = []
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, LinearWithFBS):
            res.append(name + '.fbs')
    return res


class StaticFBS(nn.Module):
    def __init__(self, w, window_merge=None):
        super(StaticFBS, self).__init__()
        assert w.dim() == 2 and w.size(0) == 1
        self.w = nn.Parameter(w, requires_grad=False) # (1, dim)
        # if window_merge is not None:
        #     self.register_buffer('window_merge', torch.tensor(window_merge, device=w.device))
        self.window_merge = window_merge
        
    def forward(self, x):
        if self.window_merge is None:
            return x * self.w.unsqueeze(1)
        # print(11, self.window_merge, x.size())
        return x * self.w.repeat(x.size(0), 1).unsqueeze(1)
    
    def __repr__(self):
        return f'StaticFBS({self.w.size(1)})'
            
            
def get_importance_values(model_with_fbs):
    res = {}
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, LinearWithFBS):
            res[name] = module.cached_raw_w.detach().cpu().numpy()
    return res


def make_divisible(v, divisor=8, min_val=None):
	"""
	This function is taken from the original tf repo.
	It ensures that all layers have a channel number that is divisible by 8
	It can be seen here:
	https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
	:param v:
	:param divisor:
	:param min_val:
	:return:
	"""
	if min_val is None:
		min_val = divisor
	new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


def svd_decompose_linear(layer: nn.Linear):
    device = layer.weight.data.device
    
    U, S, V = torch.svd(layer.weight.T)
    
    k = (layer.weight.size(0) * layer.weight.size(1)) // (layer.weight.size(0) + layer.weight.size(1))
    k = make_divisible(k)
    
    U = U[:, :k]
    S = S[:k]
    V = V[:, :k]
    
    layer1 = nn.Linear(layer.in_features, k, bias=False)
    layer2 = nn.Linear(k, layer.out_features, bias=True)
    
    layer1.weight.data.copy_(U.T)
    layer2.weight.data.copy_((torch.diag(S) @ V.t()).T)
    layer2.bias.data = layer.bias.data
    
    return nn.Sequential(layer1, layer2).to(device)
