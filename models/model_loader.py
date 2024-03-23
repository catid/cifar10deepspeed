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
    if arch == "vit_fa2":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
    if arch == "vit_pt22":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
    if arch == "vit_xformers":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
    if arch == "vit_lca":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
    if arch == "vit_local":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "mlp_dim", 256)
        define_param(params_dict, "local_window_size", 8)
        define_param(params_dict, "tile_size", 8)
        define_param(params_dict, "tile_window_size", 32)
    if arch == "soft_moe":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "dim", 512)
        define_param(params_dict, "depth", 4)
        define_param(params_dict, "heads", 6)
        define_param(params_dict, "num_experts", 4)
        define_param(params_dict, "expert_mult", 1)
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
    if arch == "based":
        define_param(params_dict, "patch_size", 4)
        define_param(params_dict, "d_model", 256)
        define_param(params_dict, "hidden_dim", 1024)
        define_param(params_dict, "kernel_size", 3)
        define_param(params_dict, "feature_dim", 8)
        define_param(params_dict, "num_key_value_heads", 6)
        define_param(params_dict, "num_heads", 6)
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

    dropout = getattr(args, 'dropout', 0.0)
    print(f"Using dropout rate of {dropout}")

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
            dropout = dropout,
            emb_dropout = dropout
        )

    if args.arch == "vit_fa2":
        from models.vit_fa2 import ViT
        return params_dict, ViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = dropout,
            emb_dropout = dropout
        )

    if args.arch == "vit_pt22":
        from models.vit_pt22 import ViT
        return params_dict, ViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = dropout,
            emb_dropout = dropout
        )

    if args.arch == "vit_xformers":
        from models.vit_xformers import ViT
        return params_dict, ViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = dropout,
            emb_dropout = dropout
        )

    if args.arch == "vit_lca":
        from models.vit_lca import ViT
        return params_dict, ViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = dropout,
            emb_dropout = dropout
        )

    if args.arch == "vit_local":
        from models.vit_local import LocalViT
        return params_dict, LocalViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            mlp_dim = params_dict["mlp_dim"],
            dropout = dropout,
            emb_dropout = dropout,
            local_window_size = params_dict["local_window_size"],
            tile_size = params_dict["tile_size"],
            tile_window_size = params_dict["tile_window_size"]
        )

    if args.arch == "soft_moe":
        from models.soft_moe import SoftMoEViT
        return params_dict, SoftMoEViT(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            dim = params_dict["dim"],
            depth = params_dict["depth"],
            heads = params_dict["heads"],
            num_experts = params_dict["num_experts"],
            expert_mult = params_dict["expert_mult"],
            dropout = dropout,
            emb_dropout = dropout
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
            dropout = dropout,
            emb_dropout = dropout
        )

    if args.arch == "s4":
        from models.s4model import S4Model
        return params_dict, S4Model(
            d_input=3,
            d_output=10,
            d_model=params_dict["d_model"],
            n_layers=params_dict["n_layers"],
            dropout=dropout,
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

    if args.arch == "based":
        from models.based_model import ViBased
        return params_dict, ViBased(
            image_size = 32,
            patch_size = params_dict["patch_size"],
            num_classes = 10,
            d_model=params_dict["d_model"],
            hidden_dim=params_dict["hidden_dim"],
            kernel_size=params_dict["kernel_size"],
            feature_dim=params_dict["feature_dim"],
            num_key_value_heads=params_dict["num_key_value_heads"],
            num_heads=params_dict["num_heads"],
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
            dropout = dropout,
            emb_dropout = dropout
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
            dropout = dropout,
            emb_dropout = dropout
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
            dropout = dropout,
            emb_dropout = dropout
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
            dropout = dropout,
            emb_dropout = dropout
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
            dropout = dropout,
            emb_dropout = dropout
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
            dropout = dropout,
            emb_dropout = dropout
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
            dropout = dropout,
            emb_dropout = dropout
        )

    raise Exception("Unrecognized model architecture: Check models/model_loader.py for defined options")
