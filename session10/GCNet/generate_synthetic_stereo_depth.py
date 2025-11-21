#!/usr/bin/env python3
"""
Generate simple synthetic datasets for:
- Eigen-style monocular depth (RGB + depth)
- DispNet-style stereo (left, right, disparity)

The scenes are 2.5D: random rectangles and circles at different depths
rendered onto a background, with a simple pinhole stereo model.

Output structure (default: data_synth/):

data_synth/
    mono/
        rgb/
            0000.png, 0001.png, ...
        depth/
            0000.png, 0001.png, ...   # depth in millimeters (uint16)
    stereo/
        left/
            0000.png, ...
        right/
            0000.png, ...
        disp/
            0000.png, ...             # disparity * 16 (uint16)
"""

import argparse
import random
from pathlib import Path

import numpy as np
import cv2


def render_random_scene(
    width: int = 192,
    height: int = 128,
    z_near: float = 2.0,
    z_far: float = 10.0,
    num_objects: int = 5,
    f: float = 200.0,
    baseline: float = 0.1,
):
    """
    Render a simple synthetic scene:

    - Background at depth z_far with dark gray color.
    - num_objects random rectangles/circles at random depths in [z_near, z_far].
    - Per-pixel depth map (z-buffer).
    - Compute disparity d = f * B / Z.
    - Warp left image horizontally to get right image (fronto-parallel approximation).

    Returns:
        left  : (H,W,3) uint8 RGB
        right : (H,W,3) uint8 RGB
        depth : (H,W)   float32, in meters
        disp  : (H,W)   float32, in pixels
    """
    # Background
    bg_color = np.array([40, 40, 40], dtype=np.uint8)
    left = np.tile(bg_color[None, None, :], (height, width, 1))
    depth = np.full((height, width), z_far, dtype=np.float32)  # start with far plane

    # Random objects with z-buffering
    for _ in range(num_objects):
        obj_depth = random.uniform(z_near, z_far)
        color = np.random.randint(80, 255, size=(3,), dtype=np.uint8)
        shape_type = random.choice(["rect", "circle"])

        if shape_type == "rect":
            # Random rectangle
            x1 = random.randint(0, width - 20)
            y1 = random.randint(0, height - 20)
            x2 = random.randint(x1 + 10, min(width, x1 + width // 3))
            y2 = random.randint(y1 + 10, min(height, y1 + height // 3))
            for y in range(y1, y2):
                for x in range(x1, x2):
                    if obj_depth < depth[y, x]:
                        depth[y, x] = obj_depth
                        left[y, x] = color
        else:
            # Random circle
            cx = random.randint(0, width - 1)
            cy = random.randint(0, height - 1)
            radius = random.randint(min(width, height) // 10, min(width, height) // 4)
            r2 = radius * radius
            for y in range(height):
                dy = y - cy
                for x in range(width):
                    dx = x - cx
                    if dx * dx + dy * dy <= r2 and obj_depth < depth[y, x]:
                        depth[y, x] = obj_depth
                        left[y, x] = color

    # Disparity from depth: d = f * B / Z
    disp = (f * baseline) / depth
    max_disp = width * 0.2  # clamp to something reasonable
    disp = np.clip(disp, 0.0, max_disp).astype(np.float32)

    # Render right image by shifting left pixels by disparity
    right = np.tile(bg_color[None, None, :], (height, width, 1))
    for y in range(height):
        for x in range(width):
            d = disp[y, x]
            xr = int(round(x - d))  # rectified stereo: shift along x
            if 0 <= xr < width:
                right[y, xr] = left[y, x]

    return left, right, depth, disp


def generate_dataset(
    out_root: str = "data_synth",
    num_scenes: int = 50,
    width: int = 192,
    height: int = 128,
    f: float = 200.0,
    baseline: float = 0.1,
):
    """
    Generate synthetic data for both:
    - Monocular depth (Eigen-style):   mono/rgb, mono/depth
    - Stereo disparity (DispNet-style): stereo/left, stereo/right, stereo/disp
    """
    root = Path(out_root)

    mono_rgb = root / "mono" / "rgb"
    mono_depth = root / "mono" / "depth"
    stereo_left = root / "stereo" / "left"
    stereo_right = root / "stereo" / "right"
    stereo_disp = root / "stereo" / "disp"

    for d in [mono_rgb, mono_depth, stereo_left, stereo_right, stereo_disp]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_scenes} scenes into {root} ...")
    for i in range(num_scenes):
        left, right, depth, disp = render_random_scene(
            width=width,
            height=height,
            f=f,
            baseline=baseline,
        )
        name = f"{i:04d}.png"

        # Save mono RGB
        cv2.imwrite(str(mono_rgb / name), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))

        # Save depth as uint16 in millimeters (for nicer ranges)
        depth_mm = (depth * 1000.0).astype(np.uint16)
        cv2.imwrite(str(mono_depth / name), depth_mm)

        # Save stereo RGB
        cv2.imwrite(str(stereo_left / name), cv2.cvtColor(left, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(stereo_right / name), cv2.cvtColor(right, cv2.COLOR_RGB2BGR))

        # Save disparity as uint16, scaled (e.g., 1/16 pixel units)
        disp_fixed = (disp * 16.0).astype(np.uint16)
        cv2.imwrite(str(stereo_disp / name), disp_fixed)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Generated scene {i+1}/{num_scenes}")

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic stereo + depth datasets"
    )
    parser.add_argument(
        "--out_root", type=str, default="data_synth", help="Output root directory"
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=50,
        help="Number of synthetic scenes to generate",
    )
    parser.add_argument("--width", type=int, default=192, help="Image width")
    parser.add_argument("--height", type=int, default=128, help="Image height")
    parser.add_argument(
        "--f",
        type=float,
        default=200.0,
        help="Focal length in pixels (used for disparity)",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.1,
        help="Baseline in meters (used for disparity)",
    )
    args = parser.parse_args()

    generate_dataset(
        out_root=args.out_root,
        num_scenes=args.num_scenes,
        width=args.width,
        height=args.height,
        f=args.f,
        baseline=args.baseline,
    )


if __name__ == "__main__":
    main()


"""
# 1) Make synthetic data
python generate_synthdata.py \
    --out_root data_synth \
    --num_scenes 100 \
    --width 192 --height 128

python eigen_demo.py train \
    --data_root data_synth/mono \
    --epochs 20 \
    --batch_size 4 \
    --output_dir eigen_runs
"""
