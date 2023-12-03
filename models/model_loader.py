def select_model(args):
    from models.vit_small import ViT
    return ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 512,
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
