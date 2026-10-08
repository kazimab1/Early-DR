"""Backbones: EfficientNet-B0 (default) or a ViT with retinal (RETFound) weights."""
import torch
import torch.nn as nn
import timm


def build_model(cfg, n_classes):
    name = cfg.model.backbone
    if name == "vit":
        # RETFound is a ViT-Large/16 trained at 224 px. Give its checkpoint in model.pretrained_path.
        model = timm.create_model("vit_large_patch16_224", pretrained=False, num_classes=n_classes,
                                  global_pool="avg", drop_path_rate=0.2)
        path = cfg.model.pretrained_path
        if path:
            state = torch.load(path, map_location="cpu")
            state = state.get("model", state)
            state = {k: v for k, v in state.items() if not k.startswith("head.")}
            print("Loaded retinal weights:", model.load_state_dict(state, strict=False))
        else:
            print("WARNING: ViT has no pretrained weights (set model.pretrained_path)")
        return model
    try:
        return timm.create_model(name, pretrained=cfg.model.pretrained, num_classes=n_classes)
    except Exception as err:                            # e.g. Kaggle internet switched off
        print(f"WARNING: could not load pretrained weights ({err}). Training from scratch.")
        return timm.create_model(name, pretrained=False, num_classes=n_classes)


@torch.no_grad()
def adabn(model, loader, device):
    """AdaBN: re-estimate BatchNorm statistics on unlabelled images from the new source.
    Only meaningful for models that have BatchNorm (not the ViT)."""
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return
    for m in norms:
        m.reset_running_stats()
        m.momentum = None                               # plain average over all batches
    model.train()
    for images, *_ in loader:
        model(images.to(device))
    model.eval()
