
def expand_feature_names_for_cyclic_norm(feature_names, cyclical_feature_names):
    feature_names_out = []
    for feat_name in feature_names:
        is_rel_feat = feat_name.startswith('rel_')
        is_cyclic = any((feat_name == cyc_feat) or feat_name.endswith(f"_{cyc_feat}") for cyc_feat in cyclical_feature_names)
        
        if is_cyclic and not is_rel_feat:
            feature_names_out.extend([f"{feat_name}_cos", f"{feat_name}_sin"])
        else:
            feature_names_out.append(feat_name)
    return feature_names_out

def setup_feature_names(base_global_feature_names, base_bin_feature_names, cyclical_feature_names, do_cyclical_norm):
    # Replace cyclical features with their cyclical transforms/normalizations if on  
    if do_cyclical_norm:
        global_feature_names = expand_feature_names_for_cyclic_norm(base_global_feature_names.copy(), cyclical_feature_names)
        bin_feature_names = expand_feature_names_for_cyclic_norm(base_bin_feature_names.copy(), cyclical_feature_names)
    else:
        global_feature_names = base_global_feature_names
        bin_feature_names = base_bin_feature_names
    return global_feature_names, bin_feature_names