#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import numpy as np

LAT_DIR = Path("data/processed/latents_z")
OUT_DIR = Path("data/processed/radial_detector")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPECIES = [
    "Batrachyla_leptopus",
    "Batrachyla_taeniata",
    "Pleurodema_thaul",
    "Calyptocephalella_gayi",
]

# ---------------------------
# Robust scale estimators
# ---------------------------
def mad_scale(x: np.ndarray) -> float:
    """Robust scale via MAD; consistent with Gaussian when multiplied by 1.4826."""
    x = np.asarray(x)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def iqr_scale(x: np.ndarray) -> float:
    """Robust scale via IQR; for Gaussian, std ≈ IQR/1.349."""
    q75, q25 = np.percentile(x, [75, 25])
    return (q75 - q25) / 1.349

def trimmed_std(x: np.ndarray, lo=5, hi=95) -> float:
    """Std after trimming extremes (percentiles)."""
    x = np.asarray(x)
    a, b = np.percentile(x, [lo, hi])
    xt = x[(x >= a) & (x <= b)]
    return float(np.std(xt, ddof=1)) if len(xt) > 1 else float(np.std(x, ddof=1))

# ---------------------------
# Utilities
# ---------------------------
def load_latents(split: str):
    npz = np.load(LAT_DIR / f"latents_{split}.npz", allow_pickle=True)
    labels = npz["label"].astype(str)
    z = npz["z"].astype(np.float32)
    fnames = npz["fname"].astype(str) if "fname" in npz else npz["file"].astype(str)
    return labels, z, fnames

def l2_norm_rows(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(x * x, axis=1))

def compute_centroids(labels: np.ndarray, z: np.ndarray):
    centroids = {}
    for sp in SPECIES:
        idx = np.where(labels == sp)[0]
        if len(idx) == 0:
            continue
        centroids[sp] = np.mean(z[idx], axis=0)
    return centroids

def compute_radii(labels: np.ndarray, z: np.ndarray, centroids: dict):
    radii = {}
    for sp in SPECIES:
        if sp not in centroids:
            continue
        idx = np.where(labels == sp)[0]
        dz = z[idx] - centroids[sp][None, :]
        radii[sp] = l2_norm_rows(dz)
    return radii

def choose_threshold(rhos: np.ndarray, mode="percentile", q=0.99, lam=3.0, scale="mad"):
    """
    mode:
      - 'percentile': r = Q_q(rho)
      - 'mean_plus': r = mean(rho) + lam * scale(rho)
    scale: 'mad' | 'iqr' | 'trimmed'
    """
    rhos = np.asarray(rhos)
    if mode == "percentile":
        return float(np.quantile(rhos, q))
    elif mode == "mean_plus":
        mu = float(np.mean(rhos))
        if scale == "mad":
            s = float(mad_scale(rhos))
        elif scale == "iqr":
            s = float(iqr_scale(rhos))
        else:
            s = float(trimmed_std(rhos))
        return float(mu + lam * s)
    else:
        raise ValueError("mode must be 'percentile' or 'mean_plus'")

def evaluate_split(labels: np.ndarray, z: np.ndarray, centroids: dict, thresholds: dict):
    """
    Returns:
      per_class_accept_rate, overall_accept_rate
    Acceptance rule: accept class k if ||z - mu_k|| <= r_k  (one-vs-class)
    """
    per_class = {}
    accepted_any = np.zeros(len(labels), dtype=bool)

    for sp in SPECIES:
        if sp not in centroids or sp not in thresholds:
            continue
        dz = z - centroids[sp][None, :]
        d = l2_norm_rows(dz)
        acc = (d <= thresholds[sp])
        accepted_any |= acc

        idx = (labels == sp)
        if np.sum(idx) > 0:
            per_class[sp] = float(np.mean(acc[idx]))
        else:
            per_class[sp] = None

    overall = float(np.mean(accepted_any))
    return per_class, overall

# ---------------------------
# Main
# ---------------------------
def main():
    # Hyperparameters for calibration
    THRESH_MODE = "percentile"   # 'percentile' or 'mean_plus'
    Q = 0.99                     # used if THRESH_MODE='percentile'
    LAM = 3.0                    # used if THRESH_MODE='mean_plus'
    SCALE = "mad"                # 'mad'|'iqr'|'trimmed' (only for mean_plus)

    # 1) Fit on train
    y_tr, z_tr, _ = load_latents("train")
    centroids = compute_centroids(y_tr, z_tr)
    radii = compute_radii(y_tr, z_tr, centroids)

    thresholds = {}
    scales = {}
    stats = {}

    for sp, rhos in radii.items():
        # scale estimate (for reporting + optional threshold mode)
        if SCALE == "mad":
            s = float(mad_scale(rhos))
        elif SCALE == "iqr":
            s = float(iqr_scale(rhos))
        else:
            s = float(trimmed_std(rhos))
        scales[sp] = s

        rk = choose_threshold(rhos, mode=THRESH_MODE, q=Q, lam=LAM, scale=SCALE)
        thresholds[sp] = rk

        stats[sp] = {
            "n": int(len(rhos)),
            "rho_mean": float(np.mean(rhos)),
            "rho_std": float(np.std(rhos, ddof=1)) if len(rhos) > 1 else float(np.std(rhos)),
            "rho_median": float(np.median(rhos)),
            "rho_q95": float(np.quantile(rhos, 0.95)),
            "rho_q99": float(np.quantile(rhos, 0.99)),
            "scale_hat": s,
            "r_k": rk,
        }

    # 2) Evaluate on train/val/test
    report = {}
    for split in ["train", "val", "test"]:
        y, z, _ = load_latents(split)
        per_class, overall = evaluate_split(y, z, centroids, thresholds)
        report[split] = {
            "per_class_accept_rate": per_class,
            "overall_accept_any_rate": overall,
        }

    # 3) Save detector
    detector = {
        "latent_dim": int(z_tr.shape[1]),
        "threshold_mode": THRESH_MODE,
        "q": Q,
        "lambda": LAM,
        "scale_method": SCALE,
        "centroids": {k: v.tolist() for k, v in centroids.items()},
        "thresholds": thresholds,
        "class_stats_train": stats,
        "evaluation": report,
    }

    out_json = OUT_DIR / "radial_detector.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(detector, f, indent=2)

    print("Guardado:", out_json)
    print("\nResumen (accept-any):")
    for split in ["train", "val", "test"]:
        print(f"  {split:5s}: {report[split]['overall_accept_any_rate']:.3f}")

    print("\nPer-class accept rate (val):")
    for sp in SPECIES:
        v = report["val"]["per_class_accept_rate"].get(sp, None)
        if v is None:
            print(f"  {sp:25s}: n/a")
        else:
            print(f"  {sp:25s}: {v:.3f}")

if __name__ == "__main__":
    main()