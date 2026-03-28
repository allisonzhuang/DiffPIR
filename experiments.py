"""
experiments.py — Full experiment pipeline for DiffPIR project.

Runs DiffPIR with different denoisers (DiffUNet, DRUNet), PnP algorithms (HQS, DRS),
and degradation tasks (Gaussian blur, motion blur, inpainting, SR).
Also runs DPS as a baseline. Computes PSNR, SSIM, LPIPS.
"""

import json
import os
import time
from pathlib import Path

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
from deepinv.loss.metric import PSNR, SSIM, LPIPS
from deepinv.utils.demo import load_url_image

# -- Our modules --
from configs import DiffusionConfig, SolverConfig, BlurConfig, InpaintConfig, SRConfig
from models.diffunet import DiffUNet
from models.drunet import DRUNet
from restoration.pnp import hqs_step, drs_step
from restoration.diffpir import diffpir_restore, build_noise_scheduler
from degradations.blur import BlurDegradation, BlurPnPSolver
from degradations.inpaint import InpaintingDegradation, InpaintingPnPSolver
from degradations.sr import SRDegradation, SRPnPSolver

# -- Baselines via deepinv --
from deepinv.sampling import DPS as DeepInvDPS
from deepinv.optim import optim_builder
from deepinv.optim.prior import PnP
from deepinv.optim.data_fidelity import L2
import deepinv


# ============================================================================
# Test images
# ============================================================================

# Standard test image URLs (permissive sources)
TEST_IMAGES = {
    "lenna": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
    "butterfly": "https://huggingface.co/datasets/deepinv/images/resolve/main/butterfly.png",
    "celeba": "https://huggingface.co/datasets/deepinv/images/resolve/main/celeba_example.jpg",
}

IMG_SIZE = 256


def load_test_images(device="cpu"):
    """Load test images as (B=1, 3, 256, 256) tensors in [0, 1]."""
    import requests
    from PIL import Image
    from io import BytesIO

    images = {}
    transform = T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor()])

    for name, url in TEST_IMAGES.items():
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            images[name] = x
            print(f"  Loaded {name}: {x.shape}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    return images


# ============================================================================
# Degradation setup
# ============================================================================


def setup_degradation(task, device="cpu"):
    """Create degradation operator and PnP solver for a given task.

    Returns: (degradation, pnp_solver, task_name, apply_fn)
    where apply_fn takes a [0,1] image and returns degraded image in [0,1].
    """
    if task == "gaussian_blur":
        cfg = BlurConfig("gaussian")
        deg = BlurDegradation(cfg)
        solver = BlurPnPSolver(deg.kernel.to(device))

        def apply_fn(x):
            return deg.apply(x.cpu() * 2 - 1).to(device) * 0.5 + 0.5

        return deg, solver, "Gaussian Blur", apply_fn

    elif task == "motion_blur":
        cfg = BlurConfig("motion")
        deg = BlurDegradation(cfg)
        solver = BlurPnPSolver(deg.kernel.to(device))

        def apply_fn(x):
            return deg.apply(x.cpu() * 2 - 1).to(device) * 0.5 + 0.5

        return deg, solver, "Motion Blur", apply_fn

    elif task == "inpainting_box":
        cfg = InpaintConfig(mask_type="box")
        deg = InpaintingDegradation(cfg, IMG_SIZE, IMG_SIZE)
        solver = InpaintingPnPSolver(deg.mask.to(device))

        def apply_fn(x):
            return deg.apply(x.cpu() * 2 - 1).to(device) * 0.5 + 0.5

        return deg, solver, "Inpainting (box)", apply_fn

    elif task == "inpainting_random":
        cfg = InpaintConfig(mask_type="random")
        deg = InpaintingDegradation(cfg, IMG_SIZE, IMG_SIZE)
        solver = InpaintingPnPSolver(deg.mask.to(device))

        def apply_fn(x):
            return deg.apply(x.cpu() * 2 - 1).to(device) * 0.5 + 0.5

        return deg, solver, "Inpainting (random)", apply_fn

    elif task == "sr_4x":
        cfg = SRConfig(scale_factor=4)
        deg = SRDegradation(cfg.scale_factor)
        solver = SRPnPSolver(cfg)

        def apply_fn(x):
            return deg.apply(x.cpu() * 2 - 1).to(device) * 0.5 + 0.5

        return deg, solver, "SR (4x)", apply_fn

    else:
        raise ValueError(f"Unknown task: {task}")


# ============================================================================
# Run DiffPIR
# ============================================================================


def run_diffpir(y_01, pnp_solver, prior, pnp_step_fn, solver_cfg, noise_scheduler):
    """Run DiffPIR restoration. Input/output in [0,1]."""
    y = y_01 * 2 - 1  # to [-1,1]
    with torch.no_grad():
        x_hat = diffpir_restore(solver_cfg, y, prior, pnp_solver, pnp_step_fn, noise_scheduler)
    return x_hat.clamp(-1, 1) * 0.5 + 0.5  # back to [0,1]


# ============================================================================
# Run DPS (via deepinv)
# ============================================================================


def build_deepinv_physics(task, deg, device):
    """Build a deepinv.physics operator matching our degradation for DPS."""
    if task in ("gaussian_blur", "motion_blur"):
        kernel = deg.kernel.unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        physics = deepinv.physics.BlurFFT(
            img_size=(3, IMG_SIZE, IMG_SIZE),
            filter=kernel,
            device=device,
        )
        return physics

    elif task in ("inpainting_box", "inpainting_random"):
        mask = deg.mask.to(device)
        physics = deepinv.physics.Inpainting(
            tensor_size=(3, IMG_SIZE, IMG_SIZE),
            mask=mask,
            device=device,
        )
        return physics

    elif task == "sr_4x":
        physics = deepinv.physics.Downsampling(
            img_size=(3, IMG_SIZE, IMG_SIZE),
            factor=4,
            filter="bicubic",
            device=device,
        )
        return physics

    return None


def run_dps(x_gt, physics, device, dps_model, n_steps=1000):
    """Run DPS baseline. Input/output in [0,1].

    Note: DPS internally works in [-1,1]. The physics operates in [0,1].
    We generate y from x_gt in [0,1] via the physics operator.
    """
    dps = DeepInvDPS(model=dps_model, max_iter=n_steps, device=device)

    # DPS works internally in [-1,1], so generate y in that range
    x_scaled = (x_gt.to(device) * 2 - 1)
    y_phys = physics(x_scaled)

    x_hat = dps(y_phys, physics)
    return (x_hat * 0.5 + 0.5).clamp(0, 1)


def run_dpir(x_gt, physics, device, dpir_denoiser, n_iter=100):
    """Run DPIR (PnP-HQS with DRUNet) baseline via deepinv. Input/output in [0,1]."""
    prior = PnP(denoiser=dpir_denoiser)
    data_fidelity = L2()

    # Decreasing noise levels for PnP
    import numpy as np
    sigmas = np.logspace(np.log10(0.05), np.log10(0.001), n_iter).tolist()

    model = optim_builder(
        iteration="HQS",
        max_iter=n_iter,
        prior=prior,
        data_fidelity=data_fidelity,
        params_algo={"lambda": 1.0, "stepsize": 1.0, "g_param": sigmas},
        verbose=False,
    ).to(device)

    y_phys = physics(x_gt.to(device))
    with torch.no_grad():
        x_hat = model(y_phys, physics)
    return x_hat.clamp(0, 1)


# ============================================================================
# Metrics
# ============================================================================


psnr_metric = PSNR()
ssim_metric = SSIM()
lpips_metric = LPIPS()


def compute_metrics(x_hat, x_gt):
    """Compute PSNR, SSIM, LPIPS. Both inputs in [0,1]."""
    x_hat = x_hat.clamp(0, 1).cpu()
    x_gt = x_gt.clamp(0, 1).cpu()
    return {
        "psnr": psnr_metric(x_hat, x_gt).item(),
        "ssim": ssim_metric(x_hat, x_gt).item(),
        "lpips": lpips_metric(x_hat, x_gt).item(),
    }


# ============================================================================
# Main experiment runner
# ============================================================================


def run_experiments(
    tasks=("gaussian_blur", "motion_blur", "inpainting_box", "sr_4x"),
    methods=("diffpir_hqs_diffunet", "diffpir_drs_diffunet", "dpir", "dps"),
    device="cuda",
    output_dir="outputs",
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading test images...")
    images = load_test_images(device)
    if not images:
        raise RuntimeError("No images loaded!")

    # Pre-load models
    print("Loading models...")
    models = {}
    if any("diffunet" in m for m in methods):
        models["diffunet"] = DiffUNet(device=device)
        print("  DiffUNet loaded")
    if any("drunet" in m for m in methods):
        models["drunet"] = DRUNet(device=device)
        print("  DRUNet loaded")
    if "dps" in methods:
        models["dps_denoiser"] = deepinv.models.DiffUNet(pretrained="download").to(device)
        print("  DPS denoiser loaded")
    if "dpir" in methods:
        models["dpir_denoiser"] = deepinv.models.DRUNet(pretrained="download").to(device)
        print("  DPIR denoiser loaded")

    # Noise schedule (shared)
    diff_cfg = DiffusionConfig()
    noise_scheduler = build_noise_scheduler(diff_cfg)
    solver_cfg = SolverConfig()

    # Noiseless config for inpainting
    solver_cfg_noiseless = SolverConfig(sigma_n=0.0, lambda_=1.0, zeta=0.3)

    all_results = []

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        deg, pnp_solver, task_name, apply_fn = setup_degradation(task, device)
        cfg = solver_cfg_noiseless if "inpaint" in task else solver_cfg

        # Build deepinv physics for DPS/DPIR
        di_physics = None
        if any(m in methods for m in ("dps", "dpir")):
            try:
                di_physics = build_deepinv_physics(task, deg, device)
            except Exception as e:
                print(f"  Could not build deepinv physics: {e}")

        for img_name, x_gt in images.items():
            print(f"\n  Image: {img_name}")

            # Apply degradation
            y = apply_fn(x_gt)

            # Save ground truth and degraded
            task_dir = Path(output_dir) / task / img_name
            task_dir.mkdir(parents=True, exist_ok=True)
            vutils.save_image(x_gt, task_dir / "ground_truth.png")

            if task == "sr_4x":
                # y is lower resolution for SR, save upsampled version for visualization
                y_vis = torch.nn.functional.interpolate(y, size=(IMG_SIZE, IMG_SIZE), mode="bicubic")
                vutils.save_image(y_vis.clamp(0, 1), task_dir / "degraded.png")
            else:
                vutils.save_image(y.clamp(0, 1), task_dir / "degraded.png")

            for method in methods:
                print(f"    Method: {method}...", end=" ", flush=True)
                t0 = time.time()

                try:
                    if method == "diffpir_hqs_diffunet":
                        x_hat = run_diffpir(y, pnp_solver, models["diffunet"], hqs_step, cfg, noise_scheduler)
                    elif method == "diffpir_hqs_drunet":
                        x_hat = run_diffpir(y, pnp_solver, models["drunet"], hqs_step, cfg, noise_scheduler)
                    elif method == "diffpir_drs_diffunet":
                        x_hat = run_diffpir(y, pnp_solver, models["diffunet"], drs_step, cfg, noise_scheduler)
                    elif method == "diffpir_drs_drunet":
                        x_hat = run_diffpir(y, pnp_solver, models["drunet"], drs_step, cfg, noise_scheduler)
                    elif method == "dps":
                        if di_physics is None:
                            print("SKIPPED (no physics)")
                            continue
                        x_hat = run_dps(x_gt, di_physics, device, models["dps_denoiser"], n_steps=100)
                    elif method == "dpir":
                        if di_physics is None:
                            print("SKIPPED (no physics)")
                            continue
                        x_hat = run_dpir(x_gt, di_physics, device, models["dpir_denoiser"])
                    else:
                        print(f"UNKNOWN METHOD")
                        continue

                    elapsed = time.time() - t0
                    metrics = compute_metrics(x_hat, x_gt)
                    metrics["method"] = method
                    metrics["task"] = task
                    metrics["image"] = img_name
                    metrics["time"] = elapsed
                    all_results.append(metrics)

                    print(f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} "
                          f"LPIPS={metrics['lpips']:.4f} ({elapsed:.1f}s)")

                    vutils.save_image(x_hat.clamp(0, 1).cpu(), task_dir / f"{method}.png")

                except Exception as e:
                    print(f"FAILED: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    results_path = Path(output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary table
    print_summary(all_results)

    return all_results


def print_summary(results):
    """Print a summary table of results."""
    print(f"\n{'='*80}")
    print(f"{'Method':<30} {'Task':<20} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6}")
    print(f"{'='*80}")

    # Group by task, then method — average over images
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r["method"], r["task"])
        grouped[key].append(r)

    for (method, task), entries in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        avg_psnr = sum(e["psnr"] for e in entries) / len(entries)
        avg_ssim = sum(e["ssim"] for e in entries) / len(entries)
        avg_lpips = sum(e["lpips"] for e in entries) / len(entries)
        print(f"{method:<30} {task:<20} {avg_psnr:>6.2f} {avg_ssim:>6.4f} {avg_lpips:>6.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--tasks", nargs="+",
        default=["gaussian_blur", "motion_blur", "inpainting_box", "sr_4x"],
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=["diffpir_hqs_diffunet", "diffpir_hqs_drunet", "diffpir_drs_diffunet", "dps"],
    )
    args = parser.parse_args()

    run_experiments(
        tasks=args.tasks,
        methods=args.methods,
        device=args.device,
        output_dir=args.output_dir,
    )
