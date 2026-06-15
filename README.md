# Jigsaw Puzzle Image Reconstruction
 
Reconstruction of 96×96 RGB images from 9 scrambled patches using a neural network combining a shared CNN encoder, a Transformer with soft permutation, and a per-patch CNN decoder.
 
---
 
## Problem Description
 
A 96×96 RGB image is divided into a 3×3 grid of 32×32 patches. Each patch is then center-cropped to 28×28 pixels to remove border information that would make the puzzle trivially easy. The 9 resulting patches are shuffled in a random order.
 
**Goal:** reconstruct the original 96×96 image from the 9 unordered patches, without any knowledge of the original permutation.
 
**Evaluation metric:** Mean Absolute Error (MAE) ± standard deviation, computed on the test set.
 
---
 
## Dataset
 
**STL-10** — 100,000 color images at 96×96 resolution, covering 10 classes (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck). The **unlabeled** subset is used (no class labels required).
 
| Split      | Images |
|------------|--------|
| Train      | 80,000 |
| Validation | 10,000 |
| Test       | 10,000 |
 
The dataset is downloaded automatically via `tf.keras.utils.get_file`. Saving a local copy is recommended to avoid repeated downloads.
 
---
 
## Model Architecture
 
The model (`jigsaw_inpaint`) consists of three main stages:
 
### 1. Patch Encoder (CNN)
Each 28×28 patch is zero-padded to 32×32, then processed by a shared CNN encoder (`TimeDistributed`):
- Conv + BN + ReLU → 64 channels (stride 1)
- Conv + BN + ReLU → 64 channels (stride 2, downscale to 16×16)
- Residual block → 64 channels
- Conv + BN + ReLU → 128 channels (stride 2, downscale to 8×8)
- Residual block → 128 channels
Output: feature maps `(B, 9, 8, 8, 128)` for all patches.
 
### 2. Transformer (Soft Permutation)
Patch tokens are projected into a D_MODEL=256 space and processed by 4 Transformer blocks with Multi-Head Attention (4 heads).
 
A **soft permutation** mechanism — using learnable output queries and attention scores — assigns each input patch to the correct grid cell, producing:
- Reordered spatial features: `(B, 9, 8, 8, 128)`
- Reordered context tokens: `(B, 9, 256)`
### 3. Patch Decoder + Refinement
Each patch is reconstructed at 32×32 pixels by a shared CNN decoder (upsampling 8→16→32). The 9 cells are tiled together into a 96×96 image.
 
A final **residual refinement** stage (3 conv layers) corrects seam artifacts at patch boundaries, operating in logit space and outputting the final image via sigmoid.
 
**Total trainable parameters:** < 6 million (project constraint).
 
### Architecture summary
 
```
Input: (B, 9, 28, 28, 3)
  → ZeroPad → (B, 9, 32, 32, 3)
  → Patch Encoder [shared CNN] → (B, 9, 8, 8, 128)
  → GAP + Dense → (B, 9, 256)
  → Positional Embedding
  → 4x Transformer Block
  → Soft Permutation (Query Bank + Attention)
  → Patch Decoder [shared CNN] → (B, 9, 32, 32, 3)
  → Tile Assembly → (B, 96, 96, 3)
  → Residual Refinement
Output: (B, 96, 96, 3)
```
 
---
 
## Training
 
| Hyperparameter  | Value                                      |
|-----------------|--------------------------------------------|
| Optimizer       | Adam (lr=1e-3)                             |
| Loss            | MAE                                        |
| Batch size      | 32                                         |
| Max epochs      | 30                                         |
| Early stopping  | patience=2                                 |
| LR scheduler    | ReduceLROnPlateau (factor=0.1, patience=0) |
 
Best weights are saved to `best_jigsaw_weights.weights.h5`.
 
> **Note:** the notebook also includes a combined MAE + SSIM loss (`mae_ssim_loss`, α=0.84), currently commented out.
 
---
 
## Baseline
 
As a naive baseline, the **mean patch image** approach replaces every patch with the average of all 9 input patches, then assembles the result into a 96×96 image.
 
Baseline MAE: ~0.1825
 
The model is expected to achieve a significantly lower MAE.
 
---
 
## Requirements
 
```
tensorflow >= 2.x
keras
numpy
matplotlib
```
 
The notebook is designed to run on **Google Colab** (GPU recommended).
 
---
 
## Project Constraints
 
- The solution must rely entirely on neural networks — no non-neural algorithmic components
- Pre-trained models are not allowed
- Total trainable parameters must stay below 6 million
- Implementation must use Keras and run on Google Colab
- Model weights must be available for download via `gdown`
- Submission must be a single `.ipynb` notebook file (no tar archives)
---
 
## Notebook Structure
 
| Section | Content |
|---|---|
| Setup & Dataset | STL-10 download, train/val/test split |
| Data Pipeline | `PatchGenerator` (Keras) and optimized `tf.data` pipeline |
| Visualization | Sample images and scrambled puzzle display |
| Baseline | MAE computation for the mean patch image baseline |
| Model | Architecture definition and summary |
| Training | Compilation, callbacks, training loop |
| Evaluation | MAE and std on the test set |
| Results visualization | Side-by-side Input / Reconstruction / Ground Truth |
 
 
Ricostruzione di immagini 96×96 a partire da 9 patch scrambled, usando una rete neurale con encoder CNN, Transformer e decoder per patch.
 
---
 
## Descrizione del problema
 
Data un'immagine 96×96 RGB, viene suddivisa in una griglia 3×3 di patch da 32×32 pixel. Ciascuna patch viene ritagliata al centro a 28×28 (center crop) per rimuovere informazioni sui bordi che renderebbero il puzzle troppo semplice. Le 9 patch risultanti vengono rimescolate in ordine casuale.
 
**Obiettivo:** ricostruire l'immagine originale 96×96 a partire dalle 9 patch disordinate, senza conoscere la permutazione originale.
 
**Metrica di valutazione:** Mean Absolute Error (MAE) ± deviazione standard, calcolato sul test set.
 
---
 
## Dataset
 
**STL-10** — 100.000 immagini a colori 96×96, 10 classi (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck). Si utilizza il sottoinsieme **unlabeled** (senza etichette).
 
| Split      | Immagini |
|------------|----------|
| Train      | 80.000   |
| Validation | 10.000   |
| Test       | 10.000   |
 
Il dataset viene scaricato automaticamente tramite `tf.keras.utils.get_file`. Si consiglia di salvare una copia locale per evitare download ripetuti.
 
---
 
## Architettura del modello
 
Il modello (`jigsaw_inpaint`) è composto da tre stadi principali:
 
### 1. Patch Encoder (CNN)
Ogni patch 28×28 viene prima zero-padded a 32×32, poi processata da un encoder CNN condiviso (`TimeDistributed`):
- Conv + BN + ReLU → 64 canali (stride 1)
- Conv + BN + ReLU → 64 canali (stride 2, downscale a 16×16)
- Residual block → 64 canali
- Conv + BN + ReLU → 128 canali (stride 2, downscale a 8×8)
- Residual block → 128 canali
Output: feature map `(B, 9, 8, 8, 128)` per ogni patch.
 
### 2. Transformer (Soft Permutation)
I token delle 9 patch vengono proiettati in uno spazio D_MODEL=256 e processati da 4 blocchi Transformer con Multi-Head Attention (4 teste).
 
Un meccanismo di **soft permutation** tramite query apprendibili e attention scores assegna ogni patch alla cella di griglia corretta, producendo:
- Feature spaziali riordinate: `(B, 9, 8, 8, 128)`
- Token di contesto riordinati: `(B, 9, 256)`
### 3. Patch Decoder + Refinement
Ogni patch viene ricostruita in 32×32 pixel dal decoder CNN (upsampling 8→16→32). Le 9 celle vengono assemblate in un'immagine 96×96 (`tile`).
 
Un ultimo stadio di **refinement residuale** (3 conv layers) corregge i bordi tra le patch, operando in logit space e outputtando l'immagine finale tramite sigmoid.
 
**Numero di parametri:** < 6 milioni (vincolo di progetto).
 
### Schema riassuntivo
 
```
Input: (B, 9, 28, 28, 3)
  → ZeroPad → (B, 9, 32, 32, 3)
  → Patch Encoder [shared CNN] → (B, 9, 8, 8, 128)
  → GAP + Dense → (B, 9, 256)
  → Positional Embedding
  → 4x Transformer Block
  → Soft Permutation (Query Bank + Attention)
  → Patch Decoder [shared CNN] → (B, 9, 32, 32, 3)
  → Tile Assembly → (B, 96, 96, 3)
  → Residual Refinement
Output: (B, 96, 96, 3)
```
 
---
 
## Training
 
| Iperparametro   | Valore              |
|-----------------|---------------------|
| Optimizer       | Adam (lr=1e-3)      |
| Loss            | MAE                 |
| Batch size      | 32                  |
| Epoche max      | 30                  |
| Early stopping  | patience=2          |
| LR scheduler    | ReduceLROnPlateau (factor=0.1, patience=0) |
 
I migliori pesi vengono salvati in `best_jigsaw_weights.weights.h5`.
 
> **Nota:** nel notebook è presente anche una loss combinata MAE + SSIM (`mae_ssim_loss`, α=0.84), attualmente commentata.
 
---
 
## Baseline
 
Come baseline banale si usa la **mean patch image**: la patch media delle 9 input viene replicata 9 volte e assemblata in un'immagine 96×96.
 
MAE baseline: ~0.1825
 
L'obiettivo del modello è ottenere un MAE significativamente inferiore.
 
---
 
## Requisiti
 
```
tensorflow >= 2.x
keras
numpy
matplotlib
```
 
Il notebook è progettato per girare su **Google Colab** (GPU raccomandata).
 
---
 
## Vincoli di progetto
 
- Soluzione basata esclusivamente su reti neurali (niente componenti algoritmici non-neurali)
- Nessun modello pre-addestrato
- Parametri totali < 6 milioni
- Implementazione in Keras
- I pesi devono essere disponibili per il download via `gdown`
- Submission: singolo notebook `.ipynb` (no tar files)
---
 
## Struttura del notebook
 
| Sezione | Contenuto |
|---|---|
| Setup & Dataset | Download STL-10, split train/val/test |
| Data Pipeline | `PatchGenerator` (Keras) e pipeline `tf.data` ottimizzata |
| Visualizzazione | Plot immagini originali e puzzle scrambled |
| Baseline | Calcolo MAE della mean patch image |
| Modello | Definizione e summary dell'architettura |
| Training | Compilazione, callbacks, training loop |
| Evaluation | MAE e std sul test set |
| Visualizzazione risultati | Confronto Input / Ricostruzione / Ground Truth |
 
