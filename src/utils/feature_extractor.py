# src/utils/feature_extractor.py

import torch
import numpy as np
from tqdm import tqdm


def extract_features_and_logits(
    model,
    loader,
    device: torch.device,
    target_layer=None,
):
    
    model.eval()
    model.to(device)

    all_logits, all_feats = [], []
    feature_buffer = []

    # Xác định layer hook
    if target_layer is None:
        if hasattr(model, "avgpool"):
            target_layer = model.avgpool
        elif hasattr(model, "features"):
            target_layer = model.features[-1]
        else:
            raise ValueError("Not found.")

    def hook_fn(module, input, output):
        
        feat = output
        if feat.dim() > 2:
            feat = feat.flatten(start_dim=1)
        feature_buffer.append(feat.detach())

    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting", leave=False):
           
            if len(batch) == 3:
                imgs, _, _ = batch
            else:
                imgs, _ = batch

            imgs = imgs.to(device)
            logits = model(imgs)

            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            all_logits.append(logits.cpu().numpy())
            feats = feature_buffer.pop().cpu().numpy()
            all_feats.append(feats)

    handle.remove()

    logits_arr = np.concatenate(all_logits, axis=0)
    feats_arr = np.concatenate(all_feats, axis=0)
    return logits_arr, feats_arr