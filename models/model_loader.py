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

def define_param(params_dict, key, default):
    if key not in params_dict:
        params_dict[key] = default
    else:
        params_dict[key] = cast_string_to_type(params_dict[key], default)

def params_to_string(params):
    s = ""
    for key in params:
        if s:
            s += ","
        s += key + "=" + str(params[key])
    return s

def get_model_params(arch_str, params_str):
    # Convert string to dict
    params_dict = parse_config_string(params_str)

    # Add any missing dict keys from defaults
    apply_default_model_params(arch_str, params_dict)

    return params_dict


# Define new models here:

def apply_default_model_params(arch, params_dict):
    if arch == "x_transformers":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 256)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
    if arch == "vit_tiny":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
    if arch == "vit_tiny_sparse":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
        define_param(params_dict, "in_splits", 8)
        define_param(params_dict, "out_splits", 16)
    if arch == "s4":
        define_param(params_dict, "d_model", 256)
        define_param(params_dict, "n_layers", 4)
    if arch == "mamba":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "d_model", 256)
        define_param(params_dict, "d_state", 16)
        define_param(params_dict, "d_conv", 4)
        define_param(params_dict, "expand", 2)
        define_param(params_dict, "n_layers", 4)
    if arch == "vit_bojan_flat":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
    if arch == "vit_bojan_flat_and_mlp":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
        define_param(params_dict, "mlp_size", 8)
    if arch == "vit_tiny_2ff":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 128)
    if arch == "vit_tiny_fff":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "fff_depth", 7)
        define_param(params_dict, "fff_count", 1)
    if arch == "vit_tiny_fff_fanout":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "fff_depth", 1)
        define_param(params_dict, "fff_count", 1)
        define_param(params_dict, "fff_fanout", 16)
    if arch == "vit_tiny_2fff":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "fff_depth", 7)
        define_param(params_dict, "fff_count", 1)
    if arch == "vit_tiny_fff_and_mlp":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "fff_depth", 7)
        define_param(params_dict, "fff_count", 1)
        define_param(params_dict, "mlp_size", 8)

def select_model(args):
    params_dict = get_model_params(args.arch, args.params)

    if args.arch == "x_transformers":
        from x_transformers import ViTransformerWrapper, Encoder

        return params_dict, ViTransformerWrapper(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            attn_layers = Encoder(
                dim = params_dict["dim"],
                depth = params_dict["depth"],
                heads = params_dict["heads"],
                #pre_norm = False,
                residual_attn = True,
                macaron = True,
                #attn_sparse_topk = 8,
                #ff_relu_squared = True,
                use_rmsnorm = True,
                #use_simple_rmsnorm = True,
            )
        )

    if args.arch == "vit_tiny":
        from models.vit_small import ViT
        return params_dict, ViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "vit_tiny_sparse":
        from models.vit_small_sparse import ViTSparse
        return params_dict, ViTSparse(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            in_splits = params_dict["in_splits"],
            out_splits = params_dict["out_splits"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "s4":
        from models.s4model import S4Model
        return params_dict, S4Model(
            d_input=3,
            d_output=10,
            d_model=params_dict["d_model"],
            n_layers=params_dict["n_layers"],
            dropout=0.2,
            prenorm=False
        )

    if args.arch == "mamba":
        from models.mamba_model import ViM
        return params_dict, ViM(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            d_model=params_dict["d_model"],
            d_state=params_dict["d_state"],
            d_conv=params_dict["d_conv"],
            expand=params_dict["expand"],
            n_layers=params_dict["n_layers"],
        )

    if args.arch == "vit_bojan_flat":
        from models.vit_small_bojan_flat import ViT_BojanFlat
        return params_dict, ViT_BojanFlat(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "vit_bojan_flat_and_mlp":
        from models.vit_small_bojan_flat_and_mlp import ViT_BojanFlatAndMLP
        return params_dict, ViT_BojanFlatAndMLP(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            mlp_size = params_dict["mlp_size"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "vit_tiny_2ff":
        from models.vit_small_2ff import ViT
        return params_dict, ViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "vit_tiny_fff":
        from models.vit_small_fff import ViT_FFF
        return params_dict, ViT_FFF(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            fff_depth = params_dict["fff_depth"],
            fff_count = params_dict["fff_count"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "vit_tiny_fff_fanout":
        from models.vit_small_fff_fanout import ViT_FFF_Fanout
        return params_dict, ViT_FFF_Fanout(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            fff_depth = params_dict["fff_depth"],
            fff_count = params_dict["fff_count"],
            fff_fanout = params_dict["fff_fanout"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "vit_tiny_2fff":
        from models.vit_small_2fff import ViT_2FFF
        return params_dict, ViT_2FFF(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            fff_depth = params_dict["fff_depth"],
            fff_count = params_dict["fff_count"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    if args.arch == "vit_tiny_fff_and_mlp":
        from models.vit_small_fff_and_mlp import ViT_FFFAndMLP
        return params_dict, ViT_FFFAndMLP(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            fff_depth = params_dict["fff_depth"],
            fff_count = params_dict["fff_count"],
            mlp_size = params_dict["mlp_size"],
            dropout = 0.1,
            emb_dropout = 0.1
        )

    raise Exception("Unrecognized model architecture: Check models/model_loader.py for defined options")
