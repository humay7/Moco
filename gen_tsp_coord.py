#!/usr/bin/env python3
import argparse, os
import numpy as np

def main():
    p = argparse.ArgumentParser(description="Generate random TSP coordinates in [0,1]^2 and save as .npy")
    p.add_argument("--problem_size", "-n", type=int, default=10, help="Number of nodes per TSP instance")
    p.add_argument("--num_graphs", "-g", type=int, default=100, help="How many instances to generate")
    p.add_argument("--split", "-s", default="test", help="Name used in filename (e.g., test/val/train)")
    p.add_argument("--out_dir", "-o", default="data/tsp", help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional, for reproducibility)")
    p.add_argument("--dtype", choices=["float32","float64"], default="float32", help="dtype of saved coords")
    p.add_argument("--overwrite", action="store_true", help="Overwrite if file exists")
    args = p.parse_args()

    if args.problem_size <= 0 or args.num_graphs <= 0:
        raise ValueError("problem_size and num_graphs must be positive integers.")

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    coords = np.random.uniform(0.0, 1.0, size=(args.num_graphs, args.problem_size, 2)).astype(args.dtype)

    out_path = os.path.join(args.out_dir, f"{args.split}-{args.problem_size}-coords.npy")
    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite to replace it.")

    np.save(out_path, coords)
    print(f"Saved: {out_path}  shape={coords.shape}  dtype={coords.dtype}")

if __name__ == "__main__":
    main()
