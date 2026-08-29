import torch
from torch import nn
from copy import deepcopy

from ours.utils.dl.common.model import get_model_size, get_module, set_module 


def static_prune_model_by_reducing_width(model: nn.Module,
                                         reducing_width_ratio, # 0.875
                                         ff1_layers_name, ff2_layers_name):
    
    model = deepcopy(model)
    reducing_width_ratio = 1. - reducing_width_ratio
        
    def _f(n):
        return int(n * reducing_width_ratio)
        
    def l1_max_indexes(p: torch.Tensor, dim=0):
        assert dim in [0, 1]
        assert p.dim() in [1, 2, 4]
        
        if dim == 1:
            p = p.T
        
        p_norm = p.abs().contiguous().view(p.size(0), -1).sum(dim=1)
        n = p.size(0)
        return p_norm.argsort(descending=True)[0: int(n * reducing_width_ratio)].sort()[0]
    
    assert hasattr(model, 'prune_heads'), 'the passed in model is not from Hugging Face.'

    if isinstance(model.config.num_attention_heads, int): # normal case (normal transformer, llama)
        num_attn_heads = model.config.num_attention_heads
        # assert num_attn_heads % reducing_width_ratio == 0
        
        model.prune_heads({
            layer_index: torch.randperm(num_attn_heads)[0: num_attn_heads - int(num_attn_heads * reducing_width_ratio)]
            for layer_index in range(len(ff1_layers_name))
        })
    elif isinstance(model.config.num_attention_heads, list): # swinv2 case
        depths = model.config.depths
        import numpy as np
        cum_depths = list(np.cumsum(depths))
        def get_attn_heads(layer_index):
            stage_index = 0
            while True:
                if layer_index < cum_depths[stage_index]:
                    break
                stage_index += 1
            return model.config.num_attention_heads[stage_index]

        model.prune_heads({
            layer_index: torch.randperm(get_attn_heads(layer_index))[0: get_attn_heads(layer_index) - \
                max(1, int(get_attn_heads(layer_index) * reducing_width_ratio))]
            for layer_index in range(len(ff1_layers_name))
        })
    
    for ff1_name, ff2_name in zip(ff1_layers_name, ff2_layers_name):
        if isinstance(ff1_name, list): # llama case
            ff1_names = ff1_name
            for ff1_name in ff1_names:
                fc1 = get_module(model, ff1_name)
                new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                    fc1.bias is not None, fc1.weight.device)
                indexes = l1_max_indexes(fc1.weight.data, 0)
                new_fc1.weight.data.copy_(fc1.weight.data[indexes])
                if fc1.bias is not None:
                    new_fc1.bias.data.copy_(fc1.bias.data[indexes])
                set_module(model, ff1_name, new_fc1)
                
        else:
            fc1 = get_module(model, ff1_name)
            new_fc1 = nn.Linear(fc1.in_features, _f(fc1.out_features), 
                                fc1.bias is not None, fc1.weight.device)
            indexes = l1_max_indexes(fc1.weight.data, 0)
            new_fc1.weight.data.copy_(fc1.weight.data[indexes])
            if fc1.bias is not None:
                new_fc1.bias.data.copy_(fc1.bias.data[indexes])
            set_module(model, ff1_name, new_fc1)

        fc2 = get_module(model, ff2_name)
        new_fc2 = nn.Linear(_f(fc2.in_features), fc2.out_features, 
                            fc2.bias is not None, fc2.weight.device)
        new_fc2.weight.data.copy_(fc2.weight.data[:, l1_max_indexes(fc2.weight.data, 1)])
        if fc2.bias is not None:
            new_fc2.bias.data.copy_(fc2.bias.data)
        set_module(model, ff2_name, new_fc2)
        
    return model
    
    
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


def svd_decompose_linear(layer: nn.Linear, target_compression_ratio=None):
    device = layer.weight.data.device
    raw_dtype = layer.weight.data.dtype

    U, S, V = torch.svd(layer.weight.T.to(torch.float32))
    
    if target_compression_ratio is None:
        k = (layer.weight.size(0) * layer.weight.size(1)) // (layer.weight.size(0) + layer.weight.size(1))
        k = make_divisible(k)
    else:
        k = (layer.weight.size(0) * layer.weight.size(1) * target_compression_ratio) // (layer.weight.size(0) + layer.weight.size(1))
        k = max(k, 1)
        k = int(k)
    
    U = U[:, :k]
    S = S[:k]
    V = V[:, :k]
    
    layer1 = nn.Linear(layer.in_features, k, bias=False)
    layer2 = nn.Linear(k, layer.out_features, bias=layer.bias is not None)
    
    layer1.weight.data.copy_(U.T)
    layer2.weight.data.copy_((torch.diag(S) @ V.t()).T)
    if layer.bias is not None:
        layer2.bias.data = layer.bias.data
    
    return nn.Sequential(layer1, layer2).to(device).to(raw_dtype)

    
def compose_two_sequential_linear(layers: nn.Sequential):
    layer1, layer2 = layers
    device = layer1.weight.data.device
    
    layer = nn.Linear(layer1.in_features, layer2.out_features, bias=layer2.bias is not None)
    layer.weight.data.copy_(layer2.weight.data @ layer1.weight.data)
    if layer2.bias is not None:
        layer.bias.data = layer2.bias.data
    
    return layer.to(device)


if __name__ == '__main__':
    from transformers import ViTForImageClassification, BertForSequenceClassification, AutoModelForCausalLM, LlamaForCausalLM
    from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
    # model = ViTForImageClassification.from_pretrained('dnns/ckpts/vit-b-p16-224')
    # model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-multilingual-cased')
    # model = AutoModelForCausalLM.from_pretrained("JackFram/llama-160m")
    
    # model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-multilingual-cased')
    # dummy_input = {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]])}
    # print(get_model_size(model.bert.encoder, True))
    # model(**dummy_input)
    # # exit()
    # pruned_model = static_prune_model_by_reducing_width(model, 6, 
    #                                      [f'bert.encoder.layer.{i}.intermediate.dense' for i in range(12)], 
    #                                      [f'bert.encoder.layer.{i}.output.dense' for i in range(12)])
    # print(get_model_size(pruned_model.bert.encoder, True))
    # pruned_model(**dummy_input)
    
    # model = ViTForImageClassification.from_pretrained('dnns/ckpts/vit-b-p16-224')
    # dummy_input = {'pixel_values': torch.rand((1, 3, 224, 224))}
    # print(get_model_size(model, True))
    # model(**dummy_input)
    # # exit()
    # pruned_model = static_prune_model_by_reducing_width(model, 6, 
    #                                      [f'vit.encoder.layer.{i}.intermediate.dense' for i in range(12)], 
    #                                      [f'vit.encoder.layer.{i}.output.dense' for i in range(12)])
    # print(get_model_size(pruned_model, True))
    # pruned_model(**dummy_input)
    
    class AttentionPrunableLlamaForCausalLM(LlamaForCausalLM):
        def prune_heads(self, heads_to_prune):
            for layer, heads in heads_to_prune.items():
                self.prune_heads_of_a_layer(self.model.layers[layer], heads)
            
        def prune_heads_of_a_layer(self, layer, heads) -> None:
            self = layer.self_attn
            if len(heads) == 0:
                return
            heads, index = find_pruneable_heads_and_indices(
                heads, self.num_heads, self.head_dim, set()
            )

            # Prune linear layers
            self.q_proj = prune_linear_layer(self.q_proj, index)
            self.k_proj = prune_linear_layer(self.k_proj, index)
            self.v_proj = prune_linear_layer(self.v_proj, index)
            self.o_proj = prune_linear_layer(self.o_proj, index, dim=1)

            # Update hyper params and store pruned heads
            self.num_heads = self.num_heads - len(heads)
            self.num_key_value_heads = self.num_key_value_heads - len(heads)
            self.hidden_size = self.num_heads * self.head_dim
    
    model = AttentionPrunableLlamaForCausalLM.from_pretrained("JackFram/llama-160m")
    print(model.config.num_key_value_heads, model.config.num_attention_heads)
    dummy_input = {'input_ids': torch.tensor([[ 101, 5672, 2033, 2011, 2151, 3793, 2017, 1005, 1040, 2066, 1012,  102]])}
    print(get_model_size(model.model.layers, True))
    model(**dummy_input)
    # exit()
    pruned_model = static_prune_model_by_reducing_width(model, 6, 
                                         [[f'model.layers.{i}.mlp.gate_proj', f'model.layers.{i}.mlp.up_proj'] for i in range(12)], 
                                         [f'model.layers.{i}.mlp.down_proj' for i in range(12)])
    print(get_model_size(pruned_model.model.layers, True))
    pruned_model(**dummy_input)