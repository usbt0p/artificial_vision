#!/usr/bin/env python3
"""
Eigen-style Multi-Scale Monocular Depth: Training on Synthetic Data
-------------------------------------------------------------------

Uses EigenMultiScaleDepthNet (coarse + fine) with an AlexNet backbone.

Expected dataset layout (from the synthetic generator):

data_synth/
    mono/
        rgb/
            0000.png, 0001.png, ...
        depth/
            0000.png, 0001.png, ...   # uint16 depth in millimeters

Usage example:

python eigen2014_demo.py \
    --data_root data_synth/mono \
    --epochs 20 \
    --batch_size 4 \
    --output_dir eigen_original_runs
"""

import argparse
from pathlib import Path


import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# -------------------------------------------------------------
# Eigen 2014-style network (coarse + fine)
# -------------------------------------------------------------
class EigenCoarseNet(nn.Module):
    """
    Coarse global network from Eigen et al. (2014), approximated:

    - Conv layers 1–5: AlexNet features (optionally pretrained).
    - Two fully connected layers:
        fc1: 4096 units + ReLU + dropout
        fc2: outputs coarse depth map (Hc x Wc) as a single channel.

    Outputs (B,1,Hc,Wc), interpreted as log-depth or depth.
    """
    def __init__(
        self,
        coarse_height: int = 74,
        coarse_width: int = 55,
        use_pretrained_alexnet: bool = True,
        input_size: tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.coarse_height = coarse_height
        self.coarse_width = coarse_width
        self.input_size = input_size

        alex = models.alexnet(
            weights=models.AlexNet_Weights.IMAGENET1K_V1
            if use_pretrained_alexnet
            else None
        )
        self.features = alex.features  # conv1–5

        # Infer flatten dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            feat = self.features(dummy)
            feat_dim = feat.shape[1] * feat.shape[2] * feat.shape[3]

        hidden_dim = 4096
        self.fc1 = nn.Linear(feat_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_dim, coarse_height * coarse_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,3,Ha,Wa) resized to AlexNet input size
        returns: (B,1,Hc,Wc)
        """
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        feat = F.relu(self.fc1(feat))
        feat = self.dropout(feat)
        out = self.fc2(feat)
        out = out.view(-1, 1, self.coarse_height, self.coarse_width)
        return out


class EigenFineNet(nn.Module):
    """
    Fine-scale refinement network:

    - Conv1 on RGB → ReLU → MaxPool (1/2 resolution).
    - Coarse map is resized to this resolution and concatenated.
    - Conv2, Conv3 refine.
    - Output is upsampled back to full resolution.
    """
    def __init__(self, input_size: tuple[int, int] = (224, 224)):
        super().__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(3, 63, kernel_size=9, padding=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2)

    def forward(self, x_rgb: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_rgb.shape

        f1 = F.relu(self.conv1(x_rgb))   # (B,63,H,W)
        f1p = self.pool1(f1)             # (B,63,H/2,W/2)

        coarse_resized = F.interpolate(
            coarse, size=f1p.shape[2:], mode="bilinear", align_corners=False
        )                                # (B,1,H/2,W/2)

        x_cat = torch.cat([f1p, coarse_resized], dim=1)  # (B,64,H/2,W/2)
        x = F.relu(self.conv2(x_cat))
        x = self.conv3(x)                # (B,1,H/2,W/2)

        x_up = F.interpolate(
            x, size=(H, W), mode="bilinear", align_corners=False
        )                                # (B,1,H,W)
        return x_up


class EigenMultiScaleDepthNet(nn.Module):
    """
    Full Eigen-style multi-scale depth network.
    """
    def __init__(
        self,
        coarse_height: int = 74,
        coarse_width: int = 55,
        alex_input_size: tuple[int, int] = (224, 224),
        use_pretrained_alexnet: bool = True,
    ):
        super().__init__()
        self.alex_input_size = alex_input_size

        self.coarse = EigenCoarseNet(
            coarse_height=coarse_height,
            coarse_width=coarse_width,
            use_pretrained_alexnet=use_pretrained_alexnet,
            input_size=alex_input_size,
        )
        self.fine = EigenFineNet(input_size=alex_input_size)

    def forward(self, x: torch.Tensor, return_coarse: bool = False):
        """
        x: (B,3,H,W) arbitrary resolution

        returns:
            depth_full: (B,1,H,W)
            [optional] coarse_full: (B,1,H,W)
        """
        B, C, H, W = x.shape

        # Resize input to AlexNet size
        x_resized = F.interpolate(
            x, size=self.alex_input_size, mode="bilinear", align_corners=False
        )

        # Coarse global prediction
        coarse_map = self.coarse(x_resized)      # (B,1,Hc,Wc)

        # Fine refinement at AlexNet resolution
        fine_resized = self.fine(x_resized, coarse_map)  # (B,1,Ha,Wa)

        # Upsample both to original size
        coarse_full = F.interpolate(
            coarse_map, size=(H, W), mode="bilinear", align_corners=False
        )
        depth_full = F.interpolate(
            fine_resized, size=(H, W), mode="bilinear", align_corners=False
        )

        if return_coarse:
            return depth_full, coarse_full
        return depth_full


# -------------------------------------------------------------
# Synthetic mono dataset (matches generator output)
# -------------------------------------------------------------
class MonoDepthDataset(Dataset):
    """
    For synthetic data from generate_synthetic_stereo_depth.py:

    data_synth/
        mono/
            rgb/*.png       # uint8 RGB
            depth/*.png     # uint16 depth in millimeters
    """
    def __init__(self, root: str, resize: tuple[int, int] | None = None):
        self.root = Path(root)
        self.rgb_paths = sorted((self.root / "rgb").glob("*.png"))
        self.depth_root = self.root / "depth"
        self.resize = resize

        if not self.rgb_paths:
            raise RuntimeError(f"No PNGs found in {self.root / 'rgb'}")

    def __len__(self):
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        rgb_path = self.rgb_paths[idx]
        name = rgb_path.name
        depth_path = self.depth_root / name

        # Load RGB
        rgb_bgr = cv2.imread(str(rgb_path))
        if rgb_bgr is None:
            raise RuntimeError(f"Failed to read {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load depth in millimeters (uint16) → meters float32
        depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            raise RuntimeError(f"Failed to read {depth_path}")
        depth_m = depth_mm.astype(np.float32) / 1000.0  # meters

        if self.resize is not None:
            w, h = self.resize
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            depth_m = cv2.resize(depth_m, (w, h), interpolation=cv2.INTER_NEAREST)

        # Convert to tensors
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)   # (3,H,W)
        depth_t = torch.from_numpy(depth_m).unsqueeze(0) # (1,H,W)
        return rgb_t, depth_t


# -------------------------------------------------------------
# Scale-invariant loss (Eigen et al. 2014 Eq. 4, in log-space)
# -------------------------------------------------------------
def scale_invariant_loss(pred_depth: torch.Tensor,
                         gt_depth: torch.Tensor,
                         lam: float = 0.5,
                         eps: float = 1e-6) -> torch.Tensor:
    """
    pred_depth, gt_depth: (B,1,H,W), in meters (positive).
    Loss is computed over valid pixels where gt_depth > 0.

    L_si = (1/n) sum_i d_i^2 - (lam / n^2) (sum_i d_i)^2
    where d_i = log(gt_i) - log(pred_i).
    """
    assert pred_depth.shape == gt_depth.shape
    B, C, H, W = gt_depth.shape

    # Mask invalid depth (<=0)
    mask = gt_depth > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)

    gt = gt_depth[mask]
    pred = pred_depth[mask]

    gt_clamped   = torch.clamp(gt,   min=1e-3)
    pred_clamped = torch.clamp(pred, min=1e-3)

    log_gt   = torch.log(gt_clamped)
    log_pred = torch.log(pred_clamped)
    d = log_gt - log_pred

    n = d.numel()
    term1 = (d ** 2).sum() / n
    term2 = (d.sum() ** 2) / (n ** 2)
    loss = term1 - lam * term2
    return loss


# -------------------------------------------------------------
# Training loop
# -------------------------------------------------------------
def train_eigen_multiscale(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    resize: tuple[int, int] | None = (320, 240),
    use_pretrained_alexnet: bool = True,
    output_dir: str = "eigen_original_runs",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = MonoDepthDataset(data_root, resize=resize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = EigenMultiScaleDepthNet(
        coarse_height=74,
        coarse_width=55,
        alex_input_size=(224, 224),
        use_pretrained_alexnet=use_pretrained_alexnet,
    ).to(device)

    # In the original paper, conv layers are initialized from ImageNet and then fine-tuned.
    # Here we simply fine-tune the whole model (features + fc1/fc2 + fine net).
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for rgb_t, depth_t in dataloader:
            rgb_t = rgb_t.to(device)
            depth_t = depth_t.to(device)

            optimizer.zero_grad()
            pred_depth = model(rgb_t)  # (B,1,H,W)

            loss = scale_invariant_loss(pred_depth, depth_t, lam=0.5)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(n_batches, 1)
        train_losses.append(avg_loss)
        print(f"[Epoch {epoch}/{epochs}] scale-invariant loss: {avg_loss:.4f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = out_dir / "eigen_multiscale.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # Plot loss curve
    epochs_axis = list(range(1, epochs + 1))
    plt.figure()
    plt.plot(epochs_axis, train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Scale-invariant Loss")
    plt.title("Eigen 2014-style Multi-Scale Depth Training")
    plt.grid(True)
    loss_plot_path = out_dir / "train_loss.png"
    plt.savefig(loss_plot_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

    return model_path


def colorize_depth(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """
    Colorize a depth map (H,W) into RGB using a Jet colormap.
    """
    valid = depth > 0
    if vmin is None:
        vmin = float(depth[valid].min()) if valid.any() else 0.0
    if vmax is None:
        vmax = float(depth[valid].max()) if valid.any() else 1.0

    depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_norm = np.clip(depth_norm, 0, 1)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
    return depth_color


def run_eigen_inference(
    image_path: str,
    model_path: str,
    output: str = "eigen_pred.png",
    output_coarse: str | None = None,
):
    """
    Run inference with a trained EigenMultiScaleDepthNet.

    - Loads RGB image.
    - Predicts depth (fine) and coarse map.
    - Saves side-by-side visualization: [RGB | coarse | fine].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}")
    print(f"Loading image: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = img_rgb.shape
    print(f"Image size: {H} x {W}")

    # Preprocess
    img = img_rgb.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

    # Build model (same config as in training)
    model = EigenMultiScaleDepthNet(
        coarse_height=74,
        coarse_width=55,
        alex_input_size=(224, 224),
        use_pretrained_alexnet=False,   # weights come from the checkpoint
    ).to(device)

    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    with torch.no_grad():
        img_t = img_t.to(device)
        # Get both refined and coarse predictions
        depth_full, coarse_full = model(img_t, return_coarse=True)
        depth_np = depth_full.squeeze().cpu().numpy()
        coarse_np = coarse_full.squeeze().cpu().numpy()

    # Clamp to positive for visualization
    depth_np = np.clip(depth_np, 1e-3, None)
    coarse_np = np.clip(coarse_np, 1e-3, None)

    # Colorize
    # Use same dynamic range for both so colors are comparable
    vmin = float(min(depth_np.min(), coarse_np.min()))
    vmax = float(max(depth_np.max(), coarse_np.max()))
    depth_color = colorize_depth(depth_np, vmin=vmin, vmax=vmax)
    coarse_color = colorize_depth(coarse_np, vmin=vmin, vmax=vmax)

    # Resize color maps to match original image
    coarse_color = cv2.resize(coarse_color, (W, H), interpolation=cv2.INTER_NEAREST)
    depth_color = cv2.resize(depth_color, (W, H), interpolation=cv2.INTER_NEAREST)

    # Build side-by-side visualization: [RGB | coarse | fine]
    vis = np.concatenate([img_rgb, coarse_color, depth_color], axis=1)

    # Save
    cv2.imwrite(output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to {output}")

    if output_coarse is not None:
        cv2.imwrite(output_coarse, cv2.cvtColor(coarse_color, cv2.COLOR_RGB2BGR))
        print(f"Saved coarse-only depth to {output_coarse}")


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------



def main():
    parser = argparse.ArgumentParser(
        description="Eigen 2014-style Multi-Scale Depth: Train + Inference"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ----- Train mode -----
    p_train = subparsers.add_parser("train", help="Train on synthetic mono data")
    p_train.add_argument("--data_root", type=str, required=True,
                         help="Root of mono dataset (e.g., data_synth/mono)")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--resize_w", type=int, default=320)
    p_train.add_argument("--resize_h", type=int, default=240)
    p_train.add_argument("--no_pretrained", action="store_true",
                         help="Disable pretrained AlexNet weights")
    p_train.add_argument("--output_dir", type=str, default="eigen_original_runs")

    # ----- Inference mode -----
    p_infer = subparsers.add_parser("infer", help="Run inference on a single image")
    p_infer.add_argument("--image", type=str, required=True, help="Path to RGB image")
    p_infer.add_argument("--model", type=str, required=True,
                         help="Path to trained model (e.g. eigen_original_runs/eigen_multiscale.pth)")
    p_infer.add_argument("--output", type=str, default="eigen_pred.png",
                         help="Path to save visualization")
    p_infer.add_argument("--output_coarse", type=str, default=None,
                         help="Optional path to save coarse-only depth")

    args = parser.parse_args()

    if args.mode == "train":
        resize = (args.resize_w, args.resize_h)
        use_pretrained = not args.no_pretrained
        train_eigen_multiscale(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resize=resize,
            use_pretrained_alexnet=use_pretrained,
            output_dir=args.output_dir,
        )
    elif args.mode == "infer":
        run_eigen_inference(
            image_path=args.image,
            model_path=args.model,
            output=args.output,
            output_coarse=args.output_coarse,
        )


if __name__ == "__main__":
    main()





"""
python eigen2014_demo.py train \
    --data_root data_synth/mono \
    --epochs 20 \
    --batch_size 4 \
    --output_dir eigen_original_runs

"""



""""
ls data_synth/mono/rgb
# 0000.png  0001.png  ...

python eigen2014_demo.py infer \
    --data_root data_synth/mono/rgb \
    --image data_synth/mono/rgb/0000.png \
    --model eigen_original_runs/eigen_multiscale.pth \
    --output eigen_pred_0000.png
"""