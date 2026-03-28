"""
make_figures.py — Generate comparison figure grids and metrics bar charts.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default="outputs", help="Directory with experiment outputs")
parser.add_argument("--figures-dir", type=str, default="figures", help="Directory to save figures")
parser.add_argument("--images", type=str, nargs="*", default=None,
                    help="Image names to show in grid (auto-detected if not given)")
parser.add_argument("--max-images", type=int, default=5,
                    help="Max images to show in comparison grid when auto-detecting")
args = parser.parse_args()

OUTPUT_DIR = Path(args.output_dir)
FIGURES_DIR = Path(args.figures_dir)
FIGURES_DIR.mkdir(exist_ok=True)

METHODS = ["ground_truth", "degraded", "diffpir_hqs_diffunet", "diffpir_drs_diffunet", "dpir", "dps"]
METHOD_LABELS = {
    "ground_truth": "Ground Truth",
    "degraded": "Degraded",
    "diffpir_hqs_diffunet": "DiffPIR (HQS)",
    "diffpir_drs_diffunet": "DiffPIR (DRS)",
    "dpir": "DPIR (DRUNet)",
    "dps": "DPS",
}

TASK_LABELS = {
    "gaussian_blur": "Gaussian Blur",
    "motion_blur": "Motion Blur",
    "inpainting_box": "Inpainting (Box)",
    "inpainting_random": "Inpainting (Random)",
}


def discover_images():
    """Auto-detect image names from the output directory."""
    if args.images:
        return args.images
    # Find first task directory and list subdirectories
    for task in TASK_LABELS:
        task_dir = OUTPUT_DIR / task
        if task_dir.exists():
            img_names = sorted([d.name for d in task_dir.iterdir() if d.is_dir()])
            if img_names:
                # Sample evenly if too many
                if len(img_names) > args.max_images:
                    step = len(img_names) // args.max_images
                    img_names = img_names[::step][:args.max_images]
                return img_names
    return []


IMAGES = discover_images()


def make_comparison_grid():
    """One figure per task: rows = images, cols = methods."""
    tasks = [t for t in TASK_LABELS if (OUTPUT_DIR / t).exists()]

    for task in tasks:
        fig, axes = plt.subplots(
            len(IMAGES), len(METHODS),
            figsize=(3 * len(METHODS), 3 * len(IMAGES)),
        )
        if len(IMAGES) == 1:
            axes = axes[np.newaxis, :]

        for i, img_name in enumerate(IMAGES):
            for j, method in enumerate(METHODS):
                ax = axes[i, j]
                img_path = OUTPUT_DIR / task / img_name / f"{method}.png"
                if img_path.exists():
                    img = mpimg.imread(str(img_path))
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
                if j == 0:
                    ax.set_ylabel(img_name.capitalize(), fontsize=10)

        fig.suptitle(TASK_LABELS.get(task, task), fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = FIGURES_DIR / f"comparison_{task}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def make_metrics_charts():
    """Bar charts of PSNR, SSIM, LPIPS averaged per task×method."""
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        print("No results.json found")
        return

    with open(results_path) as f:
        results = json.load(f)

    # Group by (task, method)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["task"], r["method"])].append(r)

    tasks = list(TASK_LABELS.keys())
    tasks = [t for t in tasks if any(k[0] == t for k in grouped)]
    methods = ["diffpir_hqs_diffunet", "diffpir_drs_diffunet", "dpir", "dps"]
    methods = [m for m in methods if any(k[1] == m for k in grouped)]

    for metric_name, higher_better in [("psnr", True), ("ssim", True), ("lpips", False)]:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 4), sharey=False)
        if len(tasks) == 1:
            axes = [axes]

        x = np.arange(len(methods))
        width = 0.6

        for ax_idx, task in enumerate(tasks):
            ax = axes[ax_idx]
            vals = []
            for m in methods:
                entries = grouped.get((task, m), [])
                if entries:
                    avg = np.mean([e[metric_name] for e in entries])
                else:
                    avg = 0
                vals.append(avg)

            colors = ["#2196F3", "#64B5F6", "#FF9800", "#F44336"]
            bars = ax.bar(x, vals, width, color=colors[:len(methods)])
            ax.set_xticks(x)
            ax.set_xticklabels(
                [METHOD_LABELS.get(m, m).replace(" (", "\n(") for m in methods],
                fontsize=7, rotation=0,
            )
            ax.set_title(TASK_LABELS.get(task, task), fontsize=10)

            # Add value labels
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                )

        fig.suptitle(metric_name.upper(), fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_path = FIGURES_DIR / f"metrics_{metric_name}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def make_metrics_table_latex():
    """Generate LaTeX table for the report."""
    results_path = OUTPUT_DIR / "results.json"
    if not results_path.exists():
        return

    with open(results_path) as f:
        results = json.load(f)

    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["task"], r["method"])].append(r)

    tasks = ["gaussian_blur", "motion_blur", "inpainting_box", "inpainting_random"]
    methods = ["diffpir_hqs_diffunet", "diffpir_drs_diffunet", "dpir", "dps"]

    print("\n% LaTeX table")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Quantitative comparison across degradation tasks (PSNR$\uparrow$ / SSIM$\uparrow$ / LPIPS$\downarrow$).}")
    print(r"\label{tab:results}")
    print(r"\small")
    print(r"\begin{tabular}{l" + "c" * len(tasks) + "}")
    print(r"\toprule")
    header = "Method & " + " & ".join(TASK_LABELS.get(t, t) for t in tasks) + r" \\"
    print(header)
    print(r"\midrule")

    for m in methods:
        row = METHOD_LABELS.get(m, m)
        for t in tasks:
            entries = grouped.get((t, m), [])
            if entries:
                psnr = np.mean([e["psnr"] for e in entries])
                ssim = np.mean([e["ssim"] for e in entries])
                lpips = np.mean([e["lpips"] for e in entries])
                row += f" & {psnr:.1f} / {ssim:.3f} / {lpips:.3f}"
            else:
                row += " & ---"
        row += r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    make_comparison_grid()
    make_metrics_charts()
    make_metrics_table_latex()
