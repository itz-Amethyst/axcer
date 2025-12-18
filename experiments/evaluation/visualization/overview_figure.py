import glob
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def compute_accuracy_avgs(base_path: str, paths: tuple):
    preferred_cols = ["exact_match_avg", "rouge-l_avg", "pass@1_avg"]
    results = []

    for entry in paths:
        if len(entry) < 3:
            print(f"[ERROR] invalid entry (expected 3 items): {entry}")
            results.append(0.0)
            continue

        sub_path, model1, model2 = entry
        avg_dirs = [
            os.path.join(base_path, sub_path, model1, "avg"),
            os.path.join(base_path, sub_path, model2, "avg"),
        ]

        collected_values = []

        for avg_dir in avg_dirs:
            csv_files = glob.glob(os.path.join(avg_dir, "*.csv"))
            if not csv_files:
                print(f"[WARN] No CSV files found in: {avg_dir}")
                continue

            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"[WARN] Could not read CSV {csv_path}: {e}")
                    continue

                found = False
                for col in preferred_cols:
                    if col in df.columns:
                        vals = df[col].dropna().tolist()
                        if not vals:
                            found = True
                            break
                        first_val = vals[0]
                        if first_val != 0.0:
                            print("val is, and path is", vals, csv_path)
                            file_name = Path(csv_path).stem
                            if file_name not in ["scitldr_avg", "mbpp_avg"]:
                                new_val = round(first_val * 100, 2)
                                print("new vals is", new_val)
                                collected_values.append(new_val)
                            else:
                                new_val = round(first_val, 2)
                                collected_values.append(new_val)
                        found = True
                        break
                if not found:
                    existing = ", ".join(df.columns.tolist())
                    print(f"[WARN] No preferred accuracy column in {csv_path}. Columns: {existing}")

        if collected_values:
            print("sum all is", collected_values)
            print("number of datasets are", len(collected_values))
            avg_value = sum(collected_values) / len(collected_values)
        else:
            avg_value = 0.0
            print(f"[WARN] No accuracy values found for entry: {entry} -> returning 0.0")

        results.append(avg_value)

    return results


def compute_latency_avgs(base_path: str, paths: tuple):
    """
    returns two lists (end2end_per_category, inference_per_category)
    preserving the order of `paths`.
    """
    end2end_results = []
    inf_results = []

    for entry in paths:
        if len(entry) < 3:
            print(f"[ERROR] invalid entry (expected 3 items): {entry}")
            end2end_results.append(0.0)
            inf_results.append(0.0)
            continue

        sub_path, model1, model2 = entry
        avg_dirs = [
            os.path.join(base_path, sub_path, model1, "avg"),
            os.path.join(base_path, sub_path, model2, "avg"),
        ]

        collected_end2end = []
        collected_inference = []

        for avg_dir in avg_dirs:
            csv_files = glob.glob(os.path.join(avg_dir, "*.csv"))
            if not csv_files:
                print(f"[WARN] No CSV files found in: {avg_dir}")
                continue

            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"[WARN] Could not read CSV {csv_path}: {e}")
                    continue

                if "end_to_end_time_avg" in df.columns:
                    vals = df["end_to_end_time_avg"].dropna().tolist()
                    if vals:
                        if vals[0] != 0.0:
                            collected_end2end.append(vals[0])
                        else:
                            nonzero = [v for v in vals if v != 0.0]
                            if nonzero:
                                collected_end2end.append(nonzero[0])

                if "inference_time_avg" in df.columns:
                    vals = df["inference_time_avg"].dropna().tolist()
                    if vals:
                        if vals[0] != 0.0:
                            collected_inference.append(vals[0])
                        else:
                            nonzero = [v for v in vals if v != 0.0]
                            if nonzero:
                                collected_inference.append(nonzero[0])

        print("collected are: ", collected_end2end)
        e2e_avg = sum(collected_end2end) / len(collected_end2end) if collected_end2end else 0.0
        inf_avg = sum(collected_inference) / len(collected_inference) if collected_inference else 0.0

        end2end_results.append(e2e_avg)
        inf_results.append(inf_avg)

    return end2end_results, inf_results


base_path = "/home/itz-amethyst/dev/axcer/experiments/results/"  # change if needed

paths = (
    ("lingua2", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
    ("selective_context", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
    ("axcer/with_interrogative", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
    ("original", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
)
latency_paths = (
    ("lingua2", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
    ("selective_context", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
    # ("original", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
    ("axcer/with_interrogative", "Meta-Llama-3.1-8B-Instruct", "gemma-3-12b-it"),
)

category_names = {
    "lingua2": "LLMLingua2",
    "selective_context": "Selective Context",
    "axcer/with_interrogative": "Axcer (ours)",
}
accuracy = compute_accuracy_avgs(base_path, paths)
latency_end2end, latency_inference = compute_latency_avgs(base_path, latency_paths)

categories = [p[0] for p in paths]
latency_categories = [p[0] for p in latency_paths]
n = len(categories)
latency_n = len(latency_categories)

print("Computed end-to-end latencies:", list(zip(categories, latency_end2end, strict=False)))
print("Computed inference latencies:", list(zip(categories, latency_inference, strict=False)))
print("Computed accuracies:", list(zip(categories, accuracy, strict=False)))

# category_colors = ['#8B4513', '#FFD700', '#D3D3D3', '#1E90FF', '#FFB700'][:n]
category_colors = ["#C2DDD9", "#1E90FF", "#D1AC5E", "#1E90FF", "#FFB700"][:n]
category_colors = ["#8AC987", "#F4B67E", "#B291C3", "#1E90FF", "#FFB700"][:n]
category_colors = ["#F7A1B5", "#FFE29A", "#B8E4C9"]

# Option 2: Minty & Warm Pastels
# category_colors = ["#B8E4C9", "#FFD3B6", "#CBAACB"]

# Option 4: Trendy Soft Pastels
# category_colors = ["#B0D0E8", "#F3B2A6", "#FFE5B4"]

end2end_color = "#1E90FF"
end2end_alpha = 0.35

custom_params = {"axes.spines.right": False, "axes.spines.left": False, "axes.spines.top": True}
sns.set_theme(style="ticks", rc=custom_params)

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

latency_end2end_ms = [v * 1000 for v in latency_end2end]
print(latency_end2end_ms)
latency_inference_ms = [v * 1000 for v in latency_inference]
print(latency_inference_ms)

ax1 = fig.add_subplot(gs[1, 0])
x = np.arange(latency_n)
bar_width = 0.6

e2e_bars = ax1.bar(
    x, latency_end2end_ms, width=bar_width, alpha=end2end_alpha, color=end2end_color, label="End-to-End Time (avg)"
)

inf_bars = []
for i, val in enumerate(latency_inference_ms):
    b = ax1.bar(x[i], val, width=bar_width, alpha=0.95, color=category_colors[i], linewidth=0.6)
    inf_bars.append(b)

all_lat_vals = [v for v in (latency_end2end_ms + latency_inference_ms) if v and v > 0]
if all_lat_vals:
    y_min = min(all_lat_vals) * 0.5
    y_max = max(all_lat_vals) * 1.6
    y_min = max(y_min, 1e-3)
else:
    y_min, y_max = 0.1, 1600.0

ax1.set_ylim([y_min, 1700])
ax1.tick_params(axis="y", labelsize=13)


ticks = np.arange(800, 1700, 200)
ax1.set_yticks(ticks)
ax1.set_yticklabels([f"{t:.0f}" for t in ticks])

ax1.set_ylabel("End-to-End latency (ms)", fontsize=12)
ax1.set_title("Average of End-to-End time (lower is better)", fontsize=12)
ax1.set_xticks([])
ax1.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

ax2 = fig.add_subplot(gs[1, 1])
ax2.axis("off")

category_patches = [Patch(facecolor=category_colors[i]) for i in range(latency_n)]
category_labels = list(category_names.values())
category_labels = [i + " (Inference Time)" for i in category_labels]

metric_patches = [
    Patch(facecolor=end2end_color, alpha=end2end_alpha, label="Compression Time"),
]

combined_handles = category_patches + metric_patches
combined_labels = category_labels + ["Compression Time"]

legend = ax2.legend(
    handles=combined_handles,
    labels=combined_labels,
    loc="center",
    frameon=True,
    fontsize=13,
    facecolor="white",
    edgecolor="gray",
    title_fontsize=12,
    handlelength=1.5,
    handleheight=0.8,
    borderpad=0.8,
    labelspacing=0.9,
)
legend.get_frame().set_linewidth(0.3)
legend.get_frame().set_boxstyle("square", pad=0.3)


ax3 = fig.add_subplot(gs[0, 1])
x_pos = np.arange(n)

original_idx = None
original_value = None
for i, cat in enumerate(categories):
    if cat.lower() == "original":
        original_idx = i
        original_value = accuracy[i]
        break

bars_handles = []
for i, (_, acc) in enumerate(zip(categories, accuracy, strict=False)):
    if i == original_idx:
        continue
    h = ax3.bar(x_pos[i], acc, color=category_colors[i], width=0.6)
    bars_handles.append(h)

if original_value is not None and original_value != 0.0:
    ax3.axhline(y=original_value, color="#768799", linestyle="--", linewidth=2, label="Original (avg)")

y_conn = [np.nan] * n
for i, acc in enumerate(accuracy):
    if i == original_idx:
        y_conn[i] = np.nan
    else:
        y_conn[i] = acc


ax3.set_ylabel("Accuracy (%)", fontsize=12)
ax3.set_title("Average Accuracy", fontsize=12)
ax3.set_xticks([])  # keep x labels cleaner; shown on ax1
valid_acc = [v for v in accuracy if v is not None and v != 0.0]
min_y_acc = max(0, min(valid_acc) - 5) if valid_acc else 0
ax3.set_ylim([min_y_acc, 100])
ax3.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
ax3.tick_params(axis="y", labelsize=13)  # ← increase number for bigger font

# --------------------------
# BOTTOM RIGHT: Legend for categories & original (ax4)
# --------------------------
ax4 = fig.add_subplot(gs[0, 0])
ax4.axis("off")

legend_category_elements = []
legend_labels = []

for i, cat in enumerate(categories):
    cat = category_names.get(cat, cat)
    if i == original_idx:
        legend_category_elements.append(Line2D([0], [0], color="#768799", linestyle="--", linewidth=1.4))
        legend_labels.append("Original")
    else:
        legend_category_elements.append(Patch(facecolor=category_colors[i]))
        legend_labels.append(cat)

legend = ax4.legend(
    legend_category_elements,
    legend_labels,
    loc="center",
    bbox_to_anchor=(0.45, 0.5),
    frameon=True,
    fontsize=13,
    facecolor="white",
    edgecolor="gray",
    title_fontsize=12,
    handlelength=1.5,
    handleheight=0.8,
    borderpad=0.8,
    labelspacing=0.9,
)
legend.get_frame().set_linewidth(0.3)
legend.get_frame().set_boxstyle("square", pad=0.3)

fig.suptitle("Accuracy", fontsize=15, y=0.98)
fig.text(0.5, 0.48, "Latency", ha="center", fontsize=15)

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "ytick.labelsize": 10,
        "xtick.labelsize": 10,
        "legend.fontsize": 10,
    }
)
plt.tight_layout()
# plt.savefig('/home/itz-amethyst/dev/vanish/axcer/latex/figures/academic_plot.pdf', format='pdf', dpi=700, bbox_inches='tight')
plt.show()

print("Plot saved as 'academic_plot.pdf'")
print("Computed accuracies:", list(zip(categories, accuracy, strict=False)))
print("Computed end-to-end latencies:", list(zip(categories, latency_end2end, strict=False)))
print("Computed inference latencies:", list(zip(categories, latency_inference, strict=False)))
