from typing import Optional
import torch
import tqdm
from torch import nn
from ours.utils.dl.common.model import get_model_device, get_model_latency, get_model_size, get_module, get_super_module, set_module
from ours.utils.common.log import logger


class KTakesAll(nn.Module):
    def __init__(self, k):
        super(KTakesAll, self).__init__()

        self.k = k
        
    def forward(self, g: torch.Tensor):
        k = int(g.size(1) * self.k)
        
        if k == 0: # fuck this corner case, which runs correctly under pytorch 1.7.0 but completely incorrectly under pytorch 1.10.0
            return g.unsqueeze(2).unsqueeze(3)
        
        i = (-g).topk(k, 1)[1]
        t = g.scatter(1, i, 0)
        return t.unsqueeze(2).unsqueeze(3)
    
    def __repr__(self):
        return f'KTakesAll(k={self.k})'
    

class KTakesAllLinear2D(nn.Module):
    def __init__(self, k):
        super(KTakesAllLinear2D, self).__init__()

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
        return f'KTakesAllLinear2D(k={self.k})'
    

class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
        
    def forward(self, x):
        return x.abs()
    
    
class Conv2dWithFBS(nn.Module):
    def __init__(self, raw_conv2d: nn.Conv2d, raw_bn: nn.BatchNorm2d, k: float, r: int):
        super(Conv2dWithFBS, self).__init__()
        
        self.fbs = nn.Sequential(
            Abs(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(raw_conv2d.in_channels, raw_conv2d.out_channels // r),
            nn.ReLU(),
            nn.Linear(raw_conv2d.out_channels // r, raw_conv2d.out_channels),
            nn.ReLU(),
        )
        self.k_takes_all = KTakesAll(k)
        
        self.raw_conv2d = raw_conv2d
        self.bn = raw_bn # remember to clear the original BNs in the network
        
        nn.init.constant_(self.fbs[5].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[5].weight)
        
        self.cached_raw_w = None
        self.l1_reg_of_raw_w = None
        self.cached_w = None
        
    def forward(self, x):
        raw_x = self.bn(self.raw_conv2d(x))
        raw_w = self.fbs(x)
        
        self.cached_raw_w = raw_w
        self.l1_reg_of_raw_w = raw_w.norm(1)
        
        w = self.k_takes_all(raw_w)
        self.cached_w = w
        
        return raw_x * w
    

class Linear2DWithFBS(nn.Module):
    def __init__(self, raw_linear: nn.Linear, k: float, r: int, window_merge=None):
        super(Linear2DWithFBS, self).__init__()
        
        self.window_merge = window_merge
        
        self.fbs = nn.Sequential(
            # Rearrange('b n d -> b d n'),
            Abs(),
            # nn.AdaptiveAvgPool1d(1),
            # SqueezeLast(),
            nn.Linear(raw_linear.in_features, raw_linear.out_features // r),
            nn.ReLU(),
            nn.Linear(raw_linear.out_features // r, raw_linear.out_features),
            nn.ReLU(),
        )
        nn.init.constant_(self.fbs[3].bias, 1.)
        nn.init.kaiming_normal_(self.fbs[3].weight)
            
        self.k_takes_all = KTakesAllLinear2D(k)
        
        self.raw_linear = raw_linear
        
        self.cached_raw_w = None
        self.l1_reg_of_raw_w = None
        self.cached_w = None
        
    def forward(self, x):
            
        raw_x = self.raw_linear(x)
        # print(x.size(), nn.AdaptiveAvgPool1d(1)(x).size())
        raw_w = self.fbs(x)
        
        self.cached_raw_w = raw_w
        self.l1_reg_of_raw_w = raw_w.norm(1)
        
        w = self.k_takes_all(raw_w)
        self.cached_w = w
        # print(raw_x.size(), w.size())
        res = raw_x * w
        return res


def add_FBS(model: nn.Module, init_k: float, r: int, ignore_layers=None, perf_test=True, example_input=None):
    device = get_model_device(model)
    model.eval()
    example_input = example_input.to(device)
    o1 = model(example_input)
    
    if perf_test:
        before_model_size = get_model_size(model, True)
        before_model_latency = get_model_latency(
            model, example_input, 50, device, 50)

    # clear original BNs
    num_original_bns = 0
    last_conv_name = None
    conv_bn_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv_name = name
        if isinstance(module, nn.BatchNorm2d) and (ignore_layers is not None and last_conv_name not in ignore_layers):
            num_original_bns += 1
            conv_bn_map[last_conv_name] = name
    
    num_conv = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and (ignore_layers is not None and name not in ignore_layers):
            conv2d_with_fbs = Conv2dWithFBS(module, get_module(model, conv_bn_map[name]), init_k, r).to(device)
            set_module(model, name, conv2d_with_fbs)
            num_conv += 1
            
    assert num_conv == num_original_bns
    
    for bn_layer in conv_bn_map.values():
        set_module(model, bn_layer, nn.Identity())

    o2 = model(example_input)
    error = (o1 - o2).abs().max().item()
    # assert error < 1e-6, error
    
    if perf_test:
        after_model_size = get_model_size(model, True)
        after_model_latency = get_model_latency(
            model, example_input, 50, device, 50)

        logger.info(f'raw model -> raw model w/ filter selection:\n'
                    f'model size: {before_model_size:.3f}MB -> {after_model_size:.3f}MB '
                    f'latency: {before_model_latency:.6f}s -> {after_model_latency:.6f}s')
    
    logger.debug(model)
    
    return model


def get_l1_reg_in_model(model_with_fbs):
    res = 0.
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, Conv2dWithFBS):
            res += module.l1_reg_of_raw_w
    return res


def set_sparsity(model, k):
    for name, module in model.named_modules():
        if isinstance(module, KTakesAll):
            module.k = k


def clear_cache(model_with_fbs):
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, Conv2dWithFBS):
            module.cached_raw_w = None
            module.cached_w = None
            module.l1_reg_of_raw_w = None
            
            
def get_fbs_params(model_with_fbs):
    res = []
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, Conv2dWithFBS):
            res += list(module.fbs.parameters())
    return res

def get_fbs_module_names(model_with_fbs):
    res = []
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, Conv2dWithFBS):
            res.append(name + '.fbs')
    return res


def bn_cal(model: nn.Module, train_loader, num_iters, device):
    if num_iters is None:
        return {}
    
    has_bn = False
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            has_bn = True
            break
    
    if not has_bn:
        return {}
    
    def bn_calibration_init(m):
        """ calculating post-statistics of batch normalization """
        if getattr(m, 'track_running_stats', False):
            # reset all values for post-statistics
            m.reset_running_stats()
            # set bn in training mode to update post-statistics
            m.training = True
            
    with torch.no_grad():
        model.eval()
        model.apply(bn_calibration_init)
        for _ in tqdm.tqdm(range(num_iters), desc='bn cal...', dynamic_ncols=True, leave=False):
            x, _ = next(train_loader)
            model(x.to(device))
        model.eval()
        
    bn_stats = {}
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_stats[n] = m
    return bn_stats


class StaticFBS(nn.Module):
    def __init__(self, w: torch.Tensor):
        super(StaticFBS, self).__init__()
        assert w.dim() == 1
        self.w = nn.Parameter(w.unsqueeze(0).unsqueeze(2).unsqueeze(3), requires_grad=False)
        
    def forward(self, x):
        return x * self.w
    
    
def switch_bn_stats(model, bn_stats):
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and n in bn_stats:
            set_module(model, n, bn_stats[n])
            
            
def get_importance_values(model_with_fbs):
    res = {}
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, Conv2dWithFBS):
            res[name] = module.cached_raw_w.detach().cpu().numpy()
    return res
