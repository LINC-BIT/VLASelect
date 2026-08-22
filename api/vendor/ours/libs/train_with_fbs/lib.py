from torch import nn
from ours.utils.common.log import logger


def is_transformer(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.Embedding)):
            return True
    return False


def add_FBS(model: nn.Module, init_k: float, r: int, ignore_layers=None, perf_test=True, example_input=None, window_merge=False):
    
    if is_transformer(model):
        from .lib_transformer import add_FBS
        return add_FBS(model, init_k, r, ignore_layers, False, None, window_merge)
    else:
        from .lib_cnn import add_FBS
        return add_FBS(model, init_k, r, ignore_layers, perf_test, example_input)


def get_l1_reg_in_model(model_with_fbs):
    from .lib_transformer import LinearWithFBS
    from .lib_cnn import Conv2dWithFBS
    
    res = 0.
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, (Conv2dWithFBS, LinearWithFBS)):
            res += module.l1_reg_of_raw_w
    return res


def set_sparsity(model, k):
    # from .lib_transformer import KTakesAll as K1
    # from .lib_cnn import KTakesAll as K2
    
    for name, module in model.named_modules():
        # if isinstance(module, (K1, K2)):
        if 'KTakesAll' in module.__class__.__name__:
            module.k = k
            
            
def debug_sparsity(model):
    from .lib_transformer import KTakesAll as K1
    from .lib_cnn import KTakesAll as K2
    
    logger.debug(f'DEBUG: model sparsity')
    logger.debug(f'-' * 10)
    for name, module in model.named_modules():
        if isinstance(module, (K1, K2)):
            logger.debug(f'layer {name}, sparsity {module.k:.2f}')
    logger.debug(f'-' * 10)


def clear_cache(model_with_fbs):
    from .lib_transformer import LinearWithFBS
    from .lib_cnn import Conv2dWithFBS
    
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, (Conv2dWithFBS, LinearWithFBS)):
            module.cached_raw_w = None
            module.cached_w = None
            module.l1_reg_of_raw_w = None
            
            
def get_fbs_params(model_with_fbs):
    from .lib_transformer import LinearWithFBS
    from .lib_cnn import Conv2dWithFBS
    
    res = []
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, (Conv2dWithFBS, LinearWithFBS)):
            res += list(module.fbs.parameters())
    return res


def get_fbs_module_names(model_with_fbs):
    from .lib_transformer import LinearWithFBS
    from .lib_cnn import Conv2dWithFBS
    
    res = []
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, (Conv2dWithFBS, LinearWithFBS)):
            res.append(name + '.fbs')
    return res


def bn_cal(model: nn.Module, train_loader, num_iters, device):
    if is_transformer(model):
        return {}
    from .lib_cnn import bn_cal
    return bn_cal(model, train_loader, num_iters, device)
    
    
def switch_bn_stats(model, bn_stats):
    if is_transformer(model):
        return
    from .lib_cnn import switch_bn_stats
    return switch_bn_stats(model, bn_stats)
            
            
def get_importance_values(model_with_fbs):
    from .lib_transformer import LinearWithFBS
    from .lib_cnn import Conv2dWithFBS
    
    res = {}
    for name, module in model_with_fbs.named_modules():
        if isinstance(module, (Conv2dWithFBS, LinearWithFBS)):
            res[name] = module.cached_raw_w.detach().cpu().numpy()
    return res
