# Jigsaw Puzzle Image Reconstruction
 
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
 
