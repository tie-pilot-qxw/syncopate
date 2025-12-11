#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wave-quantization theoretical SM utilization for GEMM.

Model:
- Each (M,N) tile corresponds to 1 CTA.
- Number of CTAs = ceil(n/tile_m) * ceil(n/tile_n) for square GEMM n×n×n.
- A "wave" can run up to SMs CTAs concurrently.
- Average utilization (wave-quantization only):
    util = CTAs / (ceil(CTAs/SMs) * SMs)

Notes:
- tile_k (e.g., 64) does NOT change CTA count for a fixed n×n×n; it affects per-CTA work/time.
"""

from __future__ import annotations
import argparse
import csv
import math
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class Point:
    n: int
    grid_m: int
    grid_n: int
    ctas: int
    waves: int
    avg_util: float
    last_wave_util: float


def wave_utilization_for_square_gemm(n: int, sms: int, tile_m: int, tile_n: int) -> Point:
    gm = ceil_div(n, tile_m)
    gn = ceil_div(n, tile_n)
    ctas = gm * gn
    waves = ceil_div(ctas, sms)
    avg_util = ctas / (waves * sms) if waves > 0 else 0.0
    last_wave_ctas = ctas - (waves - 1) * sms if waves > 0 else 0
    last_wave_util = last_wave_ctas / sms if sms > 0 else 0.0
    return Point(n, gm, gn, ctas, waves, avg_util, last_wave_util)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sms", type=int, default=132, help="Number of SMs (default: 132)")
    ap.add_argument("--tile-m", type=int, default=128, help="Tile M (default: 128)")
    ap.add_argument("--tile-n", type=int, default=128, help="Tile N (default: 128)")
    ap.add_argument("--tile-k", type=int, default=64, help="Tile K (unused for CTA count; default: 64)")
    ap.add_argument("--n-min", type=int, default=128, help="Sweep start n (default: 128)")
    ap.add_argument("--n-max", type=int, default=16384, help="Sweep end n inclusive (default: 16384)")
    ap.add_argument("--step", type=int, default=128, help="Sweep step (default: 128)")
    ap.add_argument("--out-csv", type=str, default="", help="Optional CSV path to write results")
    ap.add_argument("--out-png", type=str, default=".", help="Optional PNG path to save the plot")
    ap.add_argument("--show-topk", type=int, default=20, help="Print worst K points (default: 20)")
    args = ap.parse_args()

    if args.n_min <= 0 or args.n_max <= 0 or args.step <= 0:
        raise SystemExit("n-min/n-max/step must be positive.")
    if args.n_min > args.n_max:
        raise SystemExit("n-min must be <= n-max.")
    if args.sms <= 0:
        raise SystemExit("sms must be positive.")
    if args.tile_m <= 0 or args.tile_n <= 0:
        raise SystemExit("tile-m and tile-n must be positive.")

    points: List[Point] = []
    for n in range(args.n_min, args.n_max + 1, args.step):
        points.append(
            wave_utilization_for_square_gemm(
                n=n,
                sms=args.sms,
                tile_m=args.tile_m,
                tile_n=args.tile_n,
            )
        )

    # Print worst points (lowest avg utilization)
    worst = sorted(points, key=lambda p: (p.avg_util, p.n))[: max(1, args.show_topk)]
    print(f"\nWorst {len(worst)} points (lowest average utilization):")
    print(" n | grid_m x grid_n | ctas | waves | avg_util(%) | last_wave(%)")
    print("-" * 72)
    for p in worst:
        print(
            f"{p.n:5d} | {p.grid_m:6d} x {p.grid_n:<6d} | {p.ctas:4d} | {p.waves:5d} |"
            f" {p.avg_util*100:9.2f} | {p.last_wave_util*100:10.2f}"
        )

    # Optional CSV
    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "grid_m", "grid_n", "ctas", "waves", "avg_util", "last_wave_util"])
            for p in points:
                w.writerow([p.n, p.grid_m, p.grid_n, p.ctas, p.waves, p.avg_util, p.last_wave_util])
        print(f"\nWrote CSV: {args.out_csv}")

    # Plot
    xs = [p.n for p in points]
    ys = [p.avg_util * 100.0 for p in points]

    plt.figure(figsize=(10, 4.8))
    plt.plot(xs, ys)
    plt.ylim(0, 100)
    plt.xlim(0, args.n_max * 1.02)
    plt.xlabel("n (GEMM size: n×n×n)")
    plt.ylabel("Theoretical SM utilization (%)\n(wave quantization only)")
    plt.title(
        f"Wave Quantization Utilization vs n  "
        f"(SMs={args.sms}, tile={args.tile_m}×{args.tile_n}×{args.tile_k})"
    )
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if args.out_png:
        plt.savefig(args.out_png, dpi=200)
        print(f"Wrote plot: {args.out_png}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
