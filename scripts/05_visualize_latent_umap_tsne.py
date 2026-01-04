#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_visualize_latent_umap_tsne.py

Visualización HONESTA del espacio latente:
- UMAP / t-SNE solo para figuras
- centroides y radios proyectados (ilustrativos)
- TODAS las decisiones del detector ocurren en el espacio latente original (z)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.manifold import TSNE

# UMAP: a veces conviene este import (más explícito)
try:
    import umap  # requiere: pip install umap-learn
except Exception as e:
    raise ImportError(
        "No se pudo importar 'umap'. Instala con: pip install umap-learn"
    ) from e


# -----------------------------
# Paths
# -----------------------------
LATENTS_NPZ = Path("data/processed/latents_z/latents_train.npz")
RADIAL_JSON = Path("data/processed/radial_detector/radial_detector.json")
OUT_DIR     = Path("figures/latent_visualization")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
data = np.load(LATENTS_NPZ, allow_pickle=True)
Z = data["z"]                       # (N, zdim)
labels = data["label"].astype(str)  # (N,)

with open(RADIAL_JSON, "r") as f:
    detector = json.load(f)

# ✅ Formato real del radial_detector.json
# keys típicas: centroids, thresholds
species = sorted(detector["centroids"].keys())
centroids = {k: np.array(detector["centroids"][k], dtype=np.float32) for k in species}
radii     = {k: float(detector["thresholds"][k]) for k in species}

# Color map (puedes cambiar si quieres)
COLORS = {
    "Batrachyla_leptopus": "#1b9e77",
    "Batrachyla_taeniata": "#d95f02",
    "Pleurodema_thaul": "#7570b3",
    "Calyptocephalella_gayi": "#e7298a",
}

# -----------------------------
# Helpers
# -----------------------------
def sample_hypersphere(mu: np.ndarray, r: float, n: int, rng: np.random.RandomState):
    """Muestras en la hiperesfera ||z - mu|| = r (en espacio latente original)."""
    d = mu.shape[0]
    v = rng.randn(n, d).astype(np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return mu[None, :] + r * v

def plot_projection(Z2, centroids2, samples2_by_class, title, fname):
    """Grafica puntos + centroides + 'radios' proyectados (ilustrativos)."""
    plt.figure(figsize=(8, 7))

    # puntos por clase
    for sp in species:
        idx = (labels == sp)
        if np.any(idx):
            plt.scatter(
                Z2[idx, 0],
                Z2[idx, 1],
                s=10,
                alpha=0.55,
                label=sp,
                color=COLORS.get(sp, None),
            )

    # centroides + radios proyectados
    ax = plt.gca()
    for sp in species:
        mu2 = centroids2[sp]
        samp2 = samples2_by_class[sp]

        # radio proyectado "promedio" (solo ilustrativo)
        rad = float(np.mean(np.linalg.norm(samp2 - mu2[None, :], axis=1)))

        circle = plt.Circle(mu2, rad, color=COLORS.get(sp, "black"), fill=False, linewidth=2, alpha=0.9)
        ax.add_patch(circle)
        plt.scatter(mu2[0], mu2[1], marker="x", s=120, color="black", linewidths=2)

    plt.title(title)
    plt.legend(frameon=False, fontsize=9)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname, dpi=300)
    plt.close()


# -----------------------------
# Precompute centroid samples (in original latent space)
# -----------------------------
rng = np.random.RandomState(42)
NSAMPLES = 200

samples_by_class = {}
for sp in species:
    samples_by_class[sp] = sample_hypersphere(centroids[sp], radii[sp], NSAMPLES, rng)


# ============================================================
# UMAP (sí tiene transform -> ideal para "proyectar" centroides)
# ============================================================
umap_model = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    random_state=42,
)

Z_umap = umap_model.fit_transform(Z)

centroids_umap = {}
samples_umap = {}

for sp in species:
    mu = centroids[sp].reshape(1, -1)
    mu2 = umap_model.transform(mu)[0]
    centroids_umap[sp] = mu2

    samp = samples_by_class[sp]
    samp2 = umap_model.transform(samp)
    samples_umap[sp] = samp2

plot_projection(
    Z_umap,
    centroids_umap,
    samples_umap,
    "UMAP projection of VAE latent space\n(centroids and projected acceptance regions; visualization only)",
    "latent_umap.png",
)


# ============================================================
# t-SNE (NO tiene transform) -> fit 1 vez con set aumentado
# ============================================================
# Construimos un set aumentado: Z + centroides + samples
centroids_stack = np.vstack([centroids[sp] for sp in species])  # (K, zdim)
samples_stack = np.vstack([samples_by_class[sp] for sp in species])  # (K*NSAMPLES, zdim)

A = np.vstack([Z, centroids_stack, samples_stack])  # (N + K + K*NSAMPLES, zdim)

tsne_model = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
)

A_tsne = tsne_model.fit_transform(A)

N = Z.shape[0]
K = len(species)

Z_tsne = A_tsne[:N, :]
centroids_tsne_arr = A_tsne[N:N+K, :]
samples_tsne_arr = A_tsne[N+K:, :]

centroids_tsne = {sp: centroids_tsne_arr[i] for i, sp in enumerate(species)}

samples_tsne = {}
offset = 0
for sp in species:
    samples_tsne[sp] = samples_tsne_arr[offset:offset+NSAMPLES, :]
    offset += NSAMPLES

plot_projection(
    Z_tsne,
    centroids_tsne,
    samples_tsne,
    "t-SNE projection of VAE latent space\n(centroids and projected acceptance regions; visualization only)",
    "latent_tsne.png",
)

print("Figuras guardadas en:", OUT_DIR)