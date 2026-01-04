#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
07_online_inference.py

Inferencia "online" (streaming) en el espacio latente ORIGINAL z:
- Carga radial_detector.json
- Lee un stream de latents (por defecto desde .npz)
- Produce a_k(t) ∈ {0,1} por ventana y métricas útiles (none-of-the-above, multi-accept)

Uso típico (desde la raíz del repo):
  python scripts/07_online_inference.py \
    --detector data/processed/radial_detector/radial_detector.json \
    --latents  data/processed/latents_z/latents_test.npz \
    --out      data/processed/online/a_test.csv

Notas:
- Aquí NO usamos PCA/UMAP/t-SNE. Las decisiones son en z (d-dim).
- El "stream" real en producción vendrá de tu encoder VAE; este script deja lista la lógica del detector.
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_detector(detector_json: Path):
    d = json.loads(detector_json.read_text())

    # JSON real (según tu print TOP-KEYS):
    # keys: 'centroids', 'thresholds', ...
    centroids_dict = d["centroids"]   # {class: [d]}
    thresholds_dict = d["thresholds"] # {class: r}

    classes = list(centroids_dict.keys())
    C = np.stack([np.array(centroids_dict[k], dtype=float) for k in classes], axis=0)  # (K,d)
    R = np.array([float(thresholds_dict[k]) for k in classes], dtype=float)            # (K,)

    meta = {
        "latent_dim": d.get("latent_dim", None),
        "threshold_mode": d.get("threshold_mode", None),
        "q": d.get("q", None),
        "lambda": d.get("lambda", None),
        "scale_method": d.get("scale_method", None),
    }
    return classes, C, R, meta


def scores_and_activity(Z, C, R):
    """
    Z: (N,d) latents (stream de ventanas)
    C: (K,d) centroides
    R: (K,) radios

    Retorna:
      dist:   (N,K) distancias euclidianas
      A:      (N,K) actividad binaria a_k(t)
      none:   (N,) 1 si ninguna esfera acepta
      multi:  (N,) 1 si >=2 esferas aceptan
      k_hat:  (N,) índice de clase asignada (o -1 si none-of-the-above)
    """
    # dist^2: (N,K)
    diff = Z[:, None, :] - C[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist2 + 1e-12)

    A = dist <= (R[None, :] + 1e-12)  # (N,K) bool
    count = A.sum(axis=1)

    none = (count == 0)
    multi = (count >= 2)

    # Asignación: si hay múltiples aceptadas, elige la de mínima distancia (Eq. argmin s_k)
    # Si none-of-the-above => -1
    k_min = np.argmin(dist, axis=1)
    k_hat = np.where(none, -1, k_min)

    return dist, A.astype(int), none.astype(int), multi.astype(int), k_hat


def load_latents_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    Z = np.asarray(data["z"], dtype=float)

    labels = None
    if "label" in data.files:
        labels = data["label"].astype(str)

    # OJO: en tus .npz guardaste fname (no "file")
    fnames = None
    if "fname" in data.files:
        fnames = data["fname"].astype(str)

    split = None
    if "split" in data.files:
        split = data["split"].astype(str)

    return Z, labels, fnames, split


def write_csv(out_csv: Path, classes, A, none, multi, k_hat, fnames=None, true_labels=None):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = []
    if fnames is not None:
        header.append("fname")
    if true_labels is not None:
        header.append("true_label")

    header += [f"a_{c}" for c in classes]
    header += ["none_of_the_above", "multi_accept", "k_hat"]

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(A.shape[0]):
            row = []
            if fnames is not None:
                row.append(str(fnames[i]))
            if true_labels is not None:
                row.append(str(true_labels[i]))

            row += [str(int(v)) for v in A[i].tolist()]
            row += [str(int(none[i])), str(int(multi[i])), str(int(k_hat[i]))]
            f.write(",".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", type=str, required=True, help="Path a radial_detector.json")
    ap.add_argument("--latents", type=str, required=True, help="Path a latents_*.npz (stream simulado)")
    ap.add_argument("--out", type=str, default="data/processed/online/online_inference.csv", help="Salida CSV")
    ap.add_argument("--print-head", type=int, default=5, help="Imprime primeras N filas en consola")
    args = ap.parse_args()

    detector_path = Path(args.detector)
    latents_path = Path(args.latents)
    out_csv = Path(args.out)

    classes, C, R, meta = load_detector(detector_path)
    Z, true_labels, fnames, split = load_latents_npz(latents_path)

    if meta.get("latent_dim") is not None and int(meta["latent_dim"]) != Z.shape[1]:
        print(f"[WARN] latent_dim detector={meta['latent_dim']} vs Z.shape[1]={Z.shape[1]} (revisar consistencia)")

    dist, A, none, multi, k_hat = scores_and_activity(Z, C, R)

    # Resumen rápido
    N = Z.shape[0]
    print("Detector classes:", classes)
    print(f"N windows: {N}")
    print(f"accept-any        : {(1.0 - none.mean()):.4f}")
    print(f"none-of-the-above : {none.mean():.4f}")
    print(f"multi-accept (>=2): {multi.mean():.4f}")

    # Opcional: per-class accept rate (si vienen labels verdaderas)
    if true_labels is not None:
        print("\nPer-true-class accept-any:")
        for c in classes:
            idx = (true_labels == c)
            if idx.sum() == 0:
                continue
            rate = (A[idx].sum(axis=1) >= 1).mean()
            print(f"  {c:22s}: {rate:.4f} (n={idx.sum()})")

    write_csv(out_csv, classes, A, none, multi, k_hat, fnames=fnames, true_labels=true_labels)
    print("\nOK ->", out_csv)

    # Mostrar primeras filas (para ver a_k(t))
    nshow = max(0, int(args.print_head))
    if nshow > 0:
        print("\nHEAD (primeras filas):")
        for i in range(min(nshow, N)):
            prefix = ""
            if fnames is not None:
                prefix += f"{fnames[i]}  "
            if true_labels is not None:
                prefix += f"[true={true_labels[i]}]  "
            ak = {classes[j]: int(A[i, j]) for j in range(len(classes))}
            print(prefix + f"a_k={ak}  none={int(none[i])}  multi={int(multi[i])}  k_hat={int(k_hat[i])}")


if __name__ == "__main__":
    main()