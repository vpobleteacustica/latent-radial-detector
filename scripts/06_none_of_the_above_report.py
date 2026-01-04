#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_none_of_the_above_report.py

Mide none-of-the-above en el espacio latente ORIGINAL (z):
- % fuera de todas las esferas
- % dentro de al menos una esfera
- % dentro de >=2 esferas (ambigüedad)
"""

import json
import numpy as np
from pathlib import Path

RADIAL_JSON = Path("data/processed/radial_detector/radial_detector.json")
LAT_DIR     = Path("data/processed/latents_z")

SPLITS = ["train", "val", "test"]

def load_detector():
    d = json.loads(RADIAL_JSON.read_text())
    species = list(d["centroids"].keys())
    centroids = np.stack([np.array(d["centroids"][sp], dtype=float) for sp in species], axis=0)  # (K, d)
    radii = np.array([float(d["thresholds"][sp]) for sp in species], dtype=float)               # (K,)
    return species, centroids, radii

def compute_membership(Z, centroids, radii):
    # Z: (N,d), centroids: (K,d), radii: (K,)
    diff = Z[:, None, :] - centroids[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)              # (N,K)
    inside = dist2 <= (radii[None, :] ** 2)          # (N,K)
    count_inside = inside.sum(axis=1)                # (N,)
    return inside, count_inside

def main():
    species, C, R = load_detector()
    print("Detector species:", species)

    for split in SPLITS:
        p = LAT_DIR / f"latents_{split}.npz"
        if not p.exists():
            print(f"[WARN] No existe {p}")
            continue

        data = np.load(p, allow_pickle=True)
        Z = data["z"].astype(float)      # (N,d)
        y = data["label"].astype(str)    # (N,)

        inside, count_inside = compute_membership(Z, C, R)

        n = len(Z)
        accept_any = np.mean(count_inside >= 1)
        none_above = np.mean(count_inside == 0)
        multi      = np.mean(count_inside >= 2)

        print("\n=== SPLIT:", split, "N=", n, "===")
        print(f"accept-any       : {accept_any:.4f}")
        print(f"none-of-the-above: {none_above:.4f}")
        print(f"multi-accept (>=2): {multi:.4f}")

        print("\nPer-true-class none-of-the-above:")
        for sp in species:
            idx = (y == sp)
            if idx.sum() == 0:
                continue
            rate = np.mean(count_inside[idx] == 0)
            print(f"  {sp:22s}: {rate:.4f}  (n={idx.sum()})")

        # Ejemplos concretos de “none-of-the-above”
        out_idx = np.where(count_inside == 0)[0]
        if out_idx.size > 0:
            # Preferimos relpath si existe; si no, fname; si no, placeholder
            if "relpath" in data.files:
                names = data["relpath"].astype(str)
            elif "fname" in data.files:
                names = data["fname"].astype(str)
            else:
                names = np.array(["<no-name-in-npz>"] * n, dtype=str)

            print("\nEjemplos none-of-the-above (hasta 5):")
            for i in out_idx[:5]:
                print(" ", y[i], names[i])

if __name__ == "__main__":
    main()