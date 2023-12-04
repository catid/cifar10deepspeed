def parse_config_string(config_str):
    pairs = config_str.split(',')  # Split the string into key-value pairs

    config_dict = {}
    for pair in pairs:
        try:
            key, value = pair.split('=')  # Split each pair by '=' to get key and value
            config_dict[key] = value  # Convert value to integer and store in dictionary
        except:
            pass
    return config_dict

def cast_string_to_type(value, target):
    target_type = type(target)

    if target_type == int:
        return int(value)
    elif target_type == float:
        return float(value)
    elif target_type == bool:
        # Here, we assume 'true' and 'false' strings for boolean, but this can be adjusted
        return value.lower() in ['true', '1', 'yes', 'y']
    # Add more types as needed
    else:
        return target_type(value)  # For other types, use the constructor directly

def define_param(params, key, default):
    if key not in params:
        params[key] = default
    else:
        params[key] = cast_string_to_type(params[key], default)

def params_to_string(params):
    s = ""
    for key in params:
        if s:
            s += ","
        s += key + "=" + str(params[key])
    return s

def select_model(args):
    params = parse_config_string(args.params)

    if args.arch == "vit_tiny":
        define_param(params, "patch_size", 4)
        define_param(params, "dim", 512)
        define_param(params, "depth", 4)
        define_param(params, "heads", 6)
        define_param(params, "mlp_dim", 256)

        # vit_small
        from models.vit_small import ViT
        return params, ViT(
            image_size = 32,
            patch_size = params["patch_size"],
            num_classes = 10,
            dim = params["dim"],
            depth = params["depth"],
            heads = params["heads"],
            mlp_dim = params["mlp_dim"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    raise Exception("Unrecognized model architecture: Check models/model_loader.py for defined options")
