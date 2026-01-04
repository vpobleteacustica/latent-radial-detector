latent-radial-detector

Pipeline mínimo, reproducible y conceptualmente correcto para:
	1.	aprender un espacio latente global usando un Autoencoder Variacional (VAE) entrenado sobre representaciones log-Mel,
	2.	ajustar un detector radial por clase (MAP isotrópico / regiones hiperesféricas) en el espacio latente original,
	3.	detectar explícitamente eventos none-of-the-above (fuera de todas las clases),
	4.	generar visualizaciones honestas del espacio latente (UMAP / t-SNE / PCA solo para figuras),
	5.	ejecutar inferencia online, produciendo un vector booleano por ventana temporal:
a_k(t) \in \{0,1\}.

**Este repositorio NO incluye datos** (audios, features, latentes ni modelos entrenados).
Contiene únicamente scripts y documentación para reproducir el enfoque metodológico.

## Idea central (visión general)

Dado un conjunto de ventanas acústicas (por ejemplo, eventos de anfibios) representadas como espectrogramas log-Mel:
	•	Se entrena un VAE para aprender embeddings latentes
z_t \in \mathbb{R}^d
	•	Para cada clase biológica k, se estima:
	•	un centroide \mu_k,
	•	un radio r_k, calculado como un cuantil q de las distancias intra-clase.
	•	Se define una regla de aceptación por clase:
a_k(t) = \mathbb{I}\left[ \lVert z_t - \mu_k \rVert \le r_k \right]
	•	Se define explícitamente la condición none-of-the-above cuando:
\sum_k a_k(t) = 0
	•	Si múltiples clases aceptan simultáneamente (\sum_k a_k(t) \ge 2), el sistema puede:
	•	reportar multi-accept (ambigüedad),
	•	o resolver por mínima distancia (opcional).

## Qué sí hace este repositorio
	•	Aprende un único espacio latente global, compartido por todas las clases.
	•	Ajusta regiones de decisión hiperesféricas bajo una aproximación Gaussiana isotrópica (MAP).
	•	Permite:
	•	cuantificar none-of-the-above,
	•	medir ambigüedad (multi-accept),
	•	ejecutar inferencia online con salidas binarias por clase.
	•	Genera figuras con UMAP, t-SNE y PCA solo para inspección cualitativa.

## Qué NO hace este repositorio (muy importante)
	•	NO toma decisiones en espacios UMAP, t-SNE o PCA.
	•	NO usa UMAP/t-SNE para definir radios, umbrales o clasificadores.
	•	NO incluye datos.

Motivo clave:
UMAP y t-SNE no preservan distancias globales de forma estable ni métrica.
Por lo tanto, cualquier radio o umbral definido en esos espacios carece de significado geométrico consistente.

**Todas las decisiones (accept / none-of-the-above / multi-accept / asignación final) se realizan exclusivamente en el espacio latente original del VAE**.

UMAP, t-SNE y PCA se utilizan solo para visualización y generación de figuras.

## Estructura de datos esperada (no incluida)

Este repositorio asume que los features log-Mel ya han sido calculados y almacenados como archivos .npy:

data/features_logmel/
  train/
    Batrachyla_leptopus/*.npy
    Batrachyla_taeniata/*.npy
    Pleurodema_thaul/*.npy
    Calyptocephalella_gayi/*.npy
  val/
    ...
  test/
    ...

**Cada archivo .npy** debe contener un arreglo float32 de forma:

(n_mels, frames)

## Internamente, los features se recortan o rellenan a un número fijo de frames (TARGET_FRAMES) para el entrenamiento.

## Los archivos de audio (WAV) no se procesan directamente aquí.
La extracción de features se asume como un paso previo.

Pipeline paso a paso (scripts 01 → 07)
	1.	01_build_logmel_dataset.py
(opcional) Construye features log-Mel a partir de audio.
	2.	02_train_vae.py
Entrena el VAE y guarda el mejor modelo.
	3.	03_extract_latents.py
Extrae embeddings latentes z (usando \mu como representación determinística).
	4.	04_fit_radial_detector.py
Ajusta centroides y radios por clase.
Guarda radial_detector.json.
	5.	05_visualize_latent_umap_tsne.py
Visualización con UMAP / t-SNE (solo figuras).
	6.	05_visualize_latent_pca.py
PCA lineal (ajustado en train, proyectando val/test).
	7.	06_none_of_the_above_report.py
Reporta none-of-the-above y multi-accept.
	8.	07_online_inference.py
Inferencia online: produce a_k(t) por ventana.

MIT License.