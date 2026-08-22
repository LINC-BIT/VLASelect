from ours.utils.common.data import flatten_2d_arr


def get_fbs_layers(qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name, only_apply_qkv=False):
    qkv_layers_name = flatten_2d_arr(qkv_layers_name)
    # ff1_layers_name = flatten_2d_arr(ff1_layers_name)
    fbs_layers = [] # [qkv].0, [proj].0, ff1 or [ff1].0
    for qkv_layer_name in qkv_layers_name:
        fbs_layers += [qkv_layer_name + '.0']
        
    if only_apply_qkv:
        return fbs_layers
    
    for proj_layer_name in proj_layers_name:
        fbs_layers += [proj_layer_name + '.0']
    for ff1_layer_name in ff1_layers_name:
        if isinstance(ff1_layer_name, list):
            for n in ff1_layer_name:
                fbs_layers += [n + '.0']
        else:
            fbs_layers += [ff1_layer_name]
    if isinstance(ff1_layers_name[0], list): # llama case
        for ff2_layer_name in ff2_layers_name:
            fbs_layers += [ff2_layer_name + '.0']
    return fbs_layers


# kb to fm
def get_matched_p_name_in_fm(kb_p_name, qkv_layers_name, proj_layers_name, ff1_layers_name, ff2_layers_name):
    if '.fbs' in kb_p_name:
        return None
    
    qkv_layers_name = flatten_2d_arr(qkv_layers_name)
    ff1_layers_name = flatten_2d_arr(ff1_layers_name)
    
    kb_module_name, p_name = '.'.join(kb_p_name.split('.')[:-1]), kb_p_name.split('.')[-1]
    
    if kb_module_name.endswith('.raw_linear'):
        kb_module_name = kb_module_name[0: -11]

    # qkv
    if kb_module_name in qkv_layers_name:
        matched_p_name = kb_module_name + '.' + p_name
    elif kb_module_name[0: -2] in qkv_layers_name:
        matched_p_name = kb_module_name[0: -2] + '.' + p_name
    
    # proj
    elif kb_module_name in proj_layers_name:
        matched_p_name = kb_module_name + '.' + p_name
    elif kb_module_name[0: -2] in proj_layers_name:
        matched_p_name = kb_module_name[0: -2] + '.' + p_name
    
    # ff1
    elif kb_module_name in ff1_layers_name:
        matched_p_name = kb_module_name + '.' + p_name
    elif kb_module_name[0: -2] in ff1_layers_name:
        matched_p_name = kb_module_name[0: -2] + '.' + p_name
        
    # ff2
    elif kb_module_name in ff2_layers_name:
        matched_p_name = kb_module_name + '.' + p_name
    elif kb_module_name[0: -2] in ff2_layers_name:
        matched_p_name = kb_module_name[0: -2] + '.' + p_name
        
    else:
        return None
    
    return matched_p_name