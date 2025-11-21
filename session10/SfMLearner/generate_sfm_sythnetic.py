#!/usr/bin/env python3
"""
Generate synthetic monocular sequences for SfMLearner-style training.

Each sequence is a simple 3D-ish scene: colored disks at random 3D positions.
A virtual pinhole camera translates horizontally, creating motion parallax.

Output structure:

out_root/
  seq_000/
    0000.png
    0001.png
    ...
  seq_001/
    ...

These sequences are compatible with MonoSequenceDataset in sfmlearner_demo.py.
"""

import argparse
from pathlib import Path
import numpy as np
import cv2


def generate_sequence(
    seq_dir: Path,
    seq_len: int = 5,
    width: int = 192,
    height: int = 128,
    num_objects: int = 30,
    min_depth: float = 1.5,
    max_depth: float = 6.0,
    cam_translation_range: float = 0.5,
    seed: int | None = None,
):
    """
    Generate one sequence of 'seq_len' frames.

    World setup:
      - Objects are colored disks with 3D positions (X,Y,Z).
      - Camera moves along X-axis from -cam_translation_range to +cam_translation_range.
      - Simple pinhole projection with intrinsics (fx, fy, cx, cy).

    We only save RGB images — no depth is required for SfMLearner training.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    seq_dir.mkdir(parents=True, exist_ok=True)

    # Camera intrinsics (roughly like in sfmlearner_demo defaults)
    fx = 0.8 * width
    fy = 0.8 * height
    cx = width / 2.0
    cy = height / 2.0

    # Sample random objects in world coordinates
    # X in [-2,2], Y in [-1,1], Z in [min_depth,max_depth]
    obj_X = rng.uniform(-2.0, 2.0, size=num_objects)
    obj_Y = rng.uniform(-1.0, 1.0, size=num_objects)
    obj_Z = rng.uniform(min_depth, max_depth, size=num_objects)

    # Random colors for each object (RGB 0–255)
    obj_colors = rng.integers(low=50, high=255, size=(num_objects, 3), dtype=np.uint8)
    # Radius can depend on depth (farther → smaller) or fixed-ish
    base_radius = min(width, height) // 20

    # Camera x-positions across the sequence
    cam_positions = np.linspace(-cam_translation_range,
                                cam_translation_range,
                                num=seq_len)

    print(f"  Generating sequence in {seq_dir} with {seq_len} frames...")

    for i, cam_x in enumerate(cam_positions):
        # Start with black background
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # For painter's algorithm: draw far objects first, near last
        order = np.argsort(obj_Z)[::-1]  # far → near
        for idx in order:
            Xw = obj_X[idx]
            Yw = obj_Y[idx]
            Zw = obj_Z[idx]

            # Camera coordinates (camera moves in X)
            Xc = Xw - cam_x
            Yc = Yw
            Zc = Zw

            if Zc <= 0.1:
                continue  # behind camera

            # Pinhole projection
            u = fx * (Xc / Zc) + cx
            v = fy * (Yc / Zc) + cy

            u_int = int(round(u))
            v_int = int(round(v))

            # Skip if off-screen
            if u_int < -50 or u_int >= width + 50 or v_int < -50 or v_int >= height + 50:
                continue

            # Depth-dependent radius (optional)
            r = int(base_radius * (min_depth / Zw))
            r = max(3, r)

            cv2.circle(img, (u_int, v_int),
                       r, color=tuple(int(c) for c in obj_colors[idx].tolist()),
                       thickness=-1)

        out_path = seq_dir / f"{i:04d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic monocular sequences for SfMLearner."
    )
    parser.add_argument("--out_root", type=str, default="data_synth/mono_seq",
                        help="Output root directory for sequences")
    parser.add_argument("--num_seqs", type=int, default=50,
                        help="Number of sequences to generate")
    parser.add_argument("--seq_len", type=int, default=5,
                        help="Number of frames per sequence")
    parser.add_argument("--width", type=int, default=192,
                        help="Image width")
    parser.add_argument("--height", type=int, default=128,
                        help="Image height")
    parser.add_argument("--num_objects", type=int, default=30,
                        help="Number of objects in each scene")
    parser.add_argument("--min_depth", type=float, default=1.5,
                        help="Minimum object depth")
    parser.add_argument("--max_depth", type=float, default=6.0,
                        help="Maximum object depth")
    parser.add_argument("--cam_translation_range", type=float, default=0.5,
                        help="Camera x-translation range across the sequence")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed base")

    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for s in range(args.num_seqs):
        seq_dir = out_root / f"seq_{s:03d}"
        generate_sequence(
            seq_dir=seq_dir,
            seq_len=args.seq_len,
            width=args.width,
            height=args.height,
            num_objects=args.num_objects,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            cam_translation_range=args.cam_translation_range,
            seed=args.seed + s,
        )

    print(f"Done! Generated {args.num_seqs} sequences in {out_root}")


if __name__ == "__main__":
    main()

"""
python generate_sfm_synthetic.py \
    --out_root data_synth/mono_seq \
    --num_seqs 100 \
    --seq_len 5 \
    --width 192 --height 128

    
python sfmlearner_demo.py train \
    --data_root data_synth/mono_seq \
    --epochs 20 \
    --batch_size 4 \
    --resize_w 192 --resize_h 128 \
    --output_dir sfmlearner_runs

python sfmlearner_demo.py infer_depth \
    --image data_synth/mono_seq/seq_000/0002.png \
    --model sfmlearner_runs/sfmlearner.pth \
    --resize_w 192 --resize_h 128 \
    --output sfm_depth_seq000_0002.png


"""
