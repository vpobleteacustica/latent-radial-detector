#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_visualize_latent_pca.py

PCA HONESTA del espacio latente:
- PCA ajustado SOLO en train
- val / test solo proyectados
- centroides y radios proyectados (ilustrativos)
- radios 2D usando el mismo percentil q del detector
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA

LAT_DIR = Path("data/processed/latents_z")
RADIAL_JSON = Path("data/processed/radial_detector/radial_detector.json")
OUT_DIR = Path("figures/latent_visualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Batrachyla_leptopus": "#1b9e77",
    "Batrachyla_taeniata": "#d95f02",
    "Pleurodema_thaul": "#7570b3",
    "Calyptocephalella_gayi": "#e7298a",
}

def load_latents(split: str):
    p = LAT_DIR / f"latents_{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"No existe: {p}")
    data = np.load(p, allow_pickle=True)
    return data["z"], data["label"]

def main():
    detector = json.loads(RADIAL_JSON.read_text())
    species = list(detector["centroids"].keys())
    centroids = {k: np.array(detector["centroids"][k], float) for k in species}
    radii = {k: float(detector["thresholds"][k]) for k in species}
    q = float(detector.get("q", 0.99))

    Z_train, y_train = load_latents("train")
    Z_val,   y_val   = load_latents("val")
    Z_test,  y_test  = load_latents("test")

    pca = PCA(n_components=2, random_state=0)
    Z2_train = pca.fit_transform(Z_train)
    Z2_val   = pca.transform(Z_val)
    Z2_test  = pca.transform(Z_test)

    evr = pca.explained_variance_ratio_
    print(f"[INFO] explained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}")
    print(f"[INFO] q usado para radios 2D = {q}")

    def project(x):
        return pca.transform(x)

    # ✅ Mejora 1: constrained_layout + un poquito más de altura
    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 7),
        sharex=True, sharey=True,
        constrained_layout=True
    )

    rng = np.random.default_rng(0)  # fijo para reproducibilidad

    for ax, split, Z2, labels in zip(
        axes,
        ["train", "val", "test"],
        [Z2_train, Z2_val, Z2_test],
        [y_train, y_val, y_test],
    ):
        # puntos
        for sp in species:
            idx = (labels == sp)
            ax.scatter(
                Z2[idx, 0],
                Z2[idx, 1],
                s=10,
                alpha=0.6,
                color=COLORS.get(sp, None),
                label=sp if split == "train" else None,
            )

        # centroides + radios proyectados (percentil q en 2D)
        for sp in species:
            mu = centroids[sp]
            r  = radii[sp]
            mu2 = project(mu.reshape(1, -1))[0]

            V = rng.normal(size=(800, mu.shape[0]))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            samples = mu + r * V
            samples2 = project(samples)

            dists = np.linalg.norm(samples2 - mu2, axis=1)
            rad2 = np.quantile(dists, q)

            circle = plt.Circle(mu2, rad2, fill=False, lw=2, color=COLORS.get(sp, "black"))
            ax.add_patch(circle)
            ax.scatter(mu2[0], mu2[1], marker="x", s=80, color="black")

        ax.set_title(
            f"PCA projection (fit on train) — {split}\n"
            f"PC1={evr[0]*100:.1f}%, PC2={evr[1]*100:.1f}% (visualization only)",
            fontsize=10
        )

        # con ejes compartidos, usar adjustable="box"
        ax.set_aspect("equal", adjustable="box")

    axes[0].legend(frameon=False, fontsize=9)

    out = OUT_DIR / "latent_pca_panels_trainfit.png"

    # ✅ Mejora 2 y 3: NO tight_layout + guardar con bbox/pad para evitar recorte
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close()

    print("OK ->", out)

if __name__ == "__main__":
    main()