import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns


# =========================
# 1. Paths and file config
# =========================
report_dir = "outputs/reports"
files = {
    "ResNet-18": "resnet18_mahalanobis_ablation_N.csv",
    "ResNet-50": "resnet50_mahalanobis_ablation_N.csv",
    "EfficientNet-B3": "effb3_mahalanobis_ablation_N.csv",
}


# =========================
# 2. Load and merge data
# =========================
all_data = []

for model_name, file_name in files.items():
    path = os.path.join(report_dir, file_name)
    if os.path.exists(path):
        df = pd.read_csv(path)

        if "N_train" in df.columns:
            df = df.rename(columns={"N_train": "N"})

        df["Backbone"] = model_name
        all_data.append(df)
    else:
        print(f"Warning: File not found -> {path}")

if not all_data:
    raise FileNotFoundError("No valid CSV files found. Please check your paths.")

df_final = pd.concat(all_data, ignore_index=True)

backbone_order = ["ResNet-18", "ResNet-50", "EfficientNet-B3"]
df_final["Backbone"] = pd.Categorical(
    df_final["Backbone"],
    categories=backbone_order,
    ordered=True
)
df_final = df_final.sort_values(["Backbone", "N"])


# =========================
# 3. Plot style
# =========================
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 400,
})

colors = {
    "ResNet-18": "#2E86DE",
    "ResNet-50": "#E67E22",
    "EfficientNet-B3": "#27AE60",
}

styles = {
    "ResNet-18": {"marker": "o"},
    "ResNet-50": {"marker": "s"},
    "EfficientNet-B3": {"marker": "^"},
}

fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.8))
ax1, ax2 = axes


# =========================
# 4. Panel (a): AUROC vs N
# =========================
for backbone in backbone_order:
    df_b = df_final[df_final["Backbone"] == backbone]
    ax1.plot(
        df_b["N"],
        df_b["AUROC"],
        label=backbone,
        color=colors[backbone],
        linewidth=2.4,
        markersize=6,
        **styles[backbone]
    )

ax1.axvspan(100, 500, color="grey", alpha=0.10, label="Low-data regime")
ax1.set_title("(a) AUROC vs. number of ID training embeddings", fontweight="bold")
ax1.set_xlabel("Number of ID training embeddings (N)")
ax1.set_ylabel("AUROC")
ax1.set_xscale("log")
ax1.set_xticks([100, 500, 1000, 2500, 5000])
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.grid(True, which="major", linestyle="--", alpha=0.28)
ax1.tick_params(axis="both", which="major", length=5, width=1)
leg1 = ax1.legend(frameon=True, fancybox=True, framealpha=0.95)
leg1.get_frame().set_edgecolor("0.8")


# =========================
# 5. Panel (b): FPR95 vs N
# =========================
for backbone in backbone_order:
    df_b = df_final[df_final["Backbone"] == backbone]
    ax2.plot(
        df_b["N"],
        df_b["FPR95"],
        label=backbone,
        color=colors[backbone],
        linewidth=2.4,
        markersize=6,
        **styles[backbone]
    )

ax2.axvspan(100, 500, color="grey", alpha=0.10)
ax2.set_title("(b) FPR95 vs. number of ID training embeddings", fontweight="bold")
ax2.set_xlabel("Number of ID training embeddings (N)")
ax2.set_ylabel("FPR95 (lower is better)")
ax2.set_xscale("log")
ax2.set_xticks([100, 500, 1000, 2500, 5000])
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.grid(True, which="major", linestyle="--", alpha=0.28)
ax2.tick_params(axis="both", which="major", length=5, width=1)
leg2 = ax2.legend(frameon=True, fancybox=True, framealpha=0.95)
leg2.get_frame().set_edgecolor("0.8")


# =========================
# 6. Final layout and save
# =========================
fig.tight_layout()

pdf_path = os.path.join(report_dir, "figure3_ablation_final.pdf")
png_path = os.path.join(report_dir, "figure3_ablation_final.png")

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, bbox_inches="tight")
plt.show()

print(f"Saved PDF: {pdf_path}")
print(f"Saved PNG: {png_path}")