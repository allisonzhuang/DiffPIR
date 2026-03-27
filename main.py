import torch
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as T
import torchvision.utils as vutils

from models.diffunet import DiffUNet
from degradations.inpaint import InpaintingDegradation, InpaintingPnPSolver
from degradations.blur import BlurDegradation, BlurPnPSolver
from restoration.pnp import hqs_step
from restoration.diffpir import diffpir_restore, build_noise_scheduler
from configs import DiffusionConfig, SolverConfig, InpaintConfig, BlurConfig

# --- Load a real image ---
url = (
    "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
)
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
img = Image.open(BytesIO(response.content)).convert("RGB").resize((256, 256))
x = T.ToTensor()(img).unsqueeze(0) * 2 - 1  # (1, 3, 256, 256), values in [-1, 1]

# --- Degrade ---
height, width = x.shape[-2:]
inpaint_cfg = InpaintConfig()
blur_cfg = BlurConfig("motion")
degradation = BlurDegradation(blur_cfg)
pnp_solver = BlurPnPSolver(degradation.kernel)
# degradation = InpaintingDegradation(inpaint_cfg, height, width)
# pnp_solver = InpaintingPnPSolver(degradation.mask)
y = degradation.apply(x)

# --- Setup ---
diff_cfg = DiffusionConfig()
noise_scheduler = build_noise_scheduler(diff_cfg)
diffpir_cfg = SolverConfig()
prior = DiffUNet()

with torch.no_grad():
    x_hat = diffpir_restore(
        diffpir_cfg, y, prior, pnp_solver, hqs_step, noise_scheduler
    )

# --- Sanity checks ---
assert x_hat.shape == x.shape, f"Shape mismatch: {x_hat.shape} vs {x.shape}"
assert not torch.isnan(x_hat).any(), "NaNs in output"
assert not torch.isinf(x_hat).any(), "Infs in output"
print(f"Output range: [{x_hat.min():.3f}, {x_hat.max():.3f}] (expected approx [-1, 1])")

# --- Visualize ---
# Convert from [-1, 1] back to [0, 1] for saving
to_01 = lambda t: t.clamp(-1, 1) * 0.5 + 0.5
vutils.save_image(to_01(x), "original.png")
vutils.save_image(to_01(y), "degraded.png")
vutils.save_image(to_01(x_hat), "restored.png")
print("Saved original.png, degraded.png, restored.png")
