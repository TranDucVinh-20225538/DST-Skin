# scripts/grad_cam.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from src.models.efficientnet_b3 import get_efficientnet_b3


# =========================
# CONFIG
# =========================
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_PATH = "data/models/efficientnet_b3_robust_best.pth"
CSV_PATH = "data/reader_study/reader_cases_selected_96.csv"
OUTPUT_PATH = "outputs/reports/figure_gradcam_reader_study_main.png"

N_PER_GROUP = 2
SEED = 42

# Global Matplotlib style (cho chữ to, dễ đọc trong paper)
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
})


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================
# MODEL LOADER
# =========================
def load_model(model_path):
    model = get_efficientnet_b3(num_classes=2, pretrained=False)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# =========================
# GRAD-CAM CORE
# =========================
def generate_gradcam(model, img_tensor, target_layer, target_class=None):
    model.eval()
    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    model.zero_grad()
    score = output[:, target_class]
    score.backward()

    fh.remove()
    bh.remove()

    activ = activations[0]
    grads = gradients[0]

    pooled_grads = torch.mean(grads, dim=(0, 2, 3))
    weighted_activ = activ.clone()

    for c in range(weighted_activ.shape[1]):
        weighted_activ[:, c, :, :] *= pooled_grads[c]

    heatmap = torch.mean(weighted_activ, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap.detach().cpu().numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, target_class


def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    h_img, w_img = orig_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w_img, h_img))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * orig_img)
    return overlay


def get_gradcam_pair(model, img_path, target_layer, target_class=None):
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.array(img_pil)

    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    heatmap, pred_class = generate_gradcam(
        model,
        img_tensor,
        target_layer,
        target_class=target_class
    )
    cam_overlay = overlay_heatmap_on_image(img_np, heatmap, alpha=0.4)

    return img_np, cam_overlay, pred_class


# =========================
# CASE SELECTION
# =========================
def select_cases_for_gradcam(csv_path, n_per_group=2, seed=42):
    df = pd.read_csv(csv_path)

    groups = [
        "SAFE_TRUSTED",
        "DANGEROUS_TRAP",
        "OOD_SHOULD_DEFER",
        "OOD_TRAP_RESCUE",
    ]

    selected = {}

    for g in groups:
        sub = df[df["behavior_group"] == g].copy()

        if len(sub) == 0:
            selected[g] = sub
            continue

        if g == "SAFE_TRUSTED":
            sub = sub.sort_values(
                ["ai_prob_pred", "dstskin_maha"],
                ascending=[False, False]
            ).head(min(n_per_group, len(sub)))

        elif g == "DANGEROUS_TRAP":
            sub = sub.sort_values(
                ["danger_score", "ai_prob_pred"],
                ascending=[False, False]
            ).head(min(n_per_group, len(sub)))

        elif g == "OOD_SHOULD_DEFER":
            sub = sub.sort_values(
                ["dstskin_maha", "ai_prob_pred"],
                ascending=[True, False]
            ).head(min(n_per_group, len(sub)))

        elif g == "OOD_TRAP_RESCUE":
            sub = sub.sort_values(
                ["danger_score", "dstskin_maha"],
                ascending=[False, True]
            ).head(min(n_per_group, len(sub)))

        selected[g] = sub.reset_index(drop=True)

    return selected


# =========================
# LABEL HELPERS
# =========================
def target_class_from_row(row):
    return 1 if row["ai_pred"] == "malignant" else 0


def short_meta(row):
    # Rút gọn, tăng fontsize được mà không bị vỡ dòng quá nhiều
    return (
        f"{row['case_id']} | GT:{row['gt_label']} | AI:{row['ai_pred']} "
        f"| p={row['ai_prob_pred']:.2f} | R:{row['reliability_flag']}"
    )


def pretty_group_name(group):
    mapping = {
        "SAFE_TRUSTED": "Safe & trusted",
        "DANGEROUS_TRAP": "Dangerous trap",
        "OOD_SHOULD_DEFER": "OOD – should defer",
        "OOD_TRAP_RESCUE": "OOD trap – rescued",
    }
    return mapping.get(group, group)


# =========================
# FIGURE BUILDER
# =========================
def build_gradcam_figure(model, selected_cases, target_layer, save_path):
    groups = [
        "SAFE_TRUSTED",
        "DANGEROUS_TRAP",
        "OOD_SHOULD_DEFER",
        "OOD_TRAP_RESCUE",
    ]

    # Phóng figure lớn hẳn
    fig, axes = plt.subplots(len(groups), 4, figsize=(16, 14))

    if len(groups) == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, group in enumerate(groups):
        group_df = selected_cases[group]

        # Tắt trục trước
        for c in range(4):
            axes[r, c].axis("off")

        for i in range(min(2, len(group_df))):
            row = group_df.iloc[i]

            img_np, cam_overlay, _ = get_gradcam_pair(
                model=model,
                img_path=row["image_path"],
                target_layer=target_layer,
                target_class=target_class_from_row(row),
            )

            col_orig = i * 2
            col_cam = i * 2 + 1

            # Original
            axes[r, col_orig].imshow(img_np)
            axes[r, col_orig].axis("off")
            axes[r, col_orig].set_title(
                short_meta(row),
                fontsize=12,   # to hẳn lên
                pad=4,
            )

            # Grad-CAM
            axes[r, col_cam].imshow(cam_overlay)
            axes[r, col_cam].axis("off")
            axes[r, col_cam].set_title(
                "Grad-CAM",
                fontsize=12,
                pad=4,
            )

        # Nhãn group
        axes[r, 0].set_ylabel(
            pretty_group_name(group),
            fontsize=13,
            fontweight="bold",
            rotation=90,
            labelpad=12,
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=400, bbox_inches="tight")
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved Grad-CAM figure to: {save_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    target_layer = model.features[-1]

    selected_cases = select_cases_for_gradcam(
        CSV_PATH,
        n_per_group=N_PER_GROUP,
        seed=SEED,
    )

    for g, df_g in selected_cases.items():
        print(f"\n=== {g} ===")
        if len(df_g) == 0:
            print("No cases found.")
        else:
            print(df_g[[
                "case_id",
                "image_path",
                "gt_label",
                "id_ood_flag",
                "ai_pred",
                "ai_prob_pred",
                "dstskin_maha",
                "reliability_flag",
                "behavior_group",
                "danger_score",
            ]])

    build_gradcam_figure(
        model=model,
        selected_cases=selected_cases,
        target_layer=target_layer,
        save_path=OUTPUT_PATH,
    )