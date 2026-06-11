# SpectralBridge: A Controlled Comparison of Audio–Visual Alignment Bridges

Can audio and images be aligned **directly**, without routing through text-mediated embedding spaces (CLIP/CLAP)? This project compares language-free encoders (Audio-MAE + DINOv2) against text-aligned encoders (CLAP + CLIP) for audio→image retrieval, across four bridge architectures trained under an identical protocol.

The proposed architecture — *SpectralBridge*, a Fourier-feature + SIREN bridge — is the hypothesis this experiment tests. The experiment rejected it. The interesting results are in what the controlled comparison revealed instead.

## Key findings

1. **A full-rank linear map is the strongest audio→visual bridge.** A single trained 768×768 linear layer between Audio-MAE and DINOv2 spaces reaches **45.7% R@5** on a held-out set where chance is 2.0%. The two encoders — trained independently, on different modalities, with no shared supervision — produce representation spaces that align to a large degree under a simple affine transform.

2. **The proposed Fourier+SIREN bridge fails to generalize, and the failure is isolated to that block.** A ReLU MLP with the same depth, width, dropout, data, and training budget generalizes well (34.3% R@5); SpectralBridge, differing only in the frozen random Fourier projection and sine activations, performs at chance level (2.8% R@5). This is a clean ablation result: the periodic-architecture hypothesis was tested and rejected on this task.

3. **Text-aligned encoders transfer poorly zero-shot but adapt cheaply.** Zero-shot CLAP→CLIP retrieval is at chance level (2.4% R@5), yet a small trained projection (132K params) lifts it to 36.2% — competitive with, though still behind, the language-free pair.

## Results

Audio→image retrieval on a held-out validation set (254 pairs, gallery = val images only):

| Method | Encoders | Trainable params | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| **Linear probe** | Audio-MAE / DINOv2 | 590,592 | **17.32** | **45.67** | **62.20** |
| CLAP→CLIP trained MLP | CLAP / CLIP | 131,712 | 14.96 | 36.22 | 57.87 |
| ReLU MLP | Audio-MAE / DINOv2 | 214,016 | 11.81 | 34.25 | 56.30 |
| SpectralBridge (proposed) | Audio-MAE / DINOv2 | 132,096 | 0.39 | 2.76 | 5.91 |
| Zero-shot CLAP→CLIP | CLAP / CLIP | 0 | 0.79 | 2.36 | 5.51 |
| *Random chance* | — | — | *0.39* | *1.97* | *3.94* |

## Dataset

1,269 audio–image pairs from a 16-category subset of [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/): the audio track and a middle video frame of each 10-second clip. Split 80/20 (1,015 train / 254 val) with a fixed seed. Categories span both "textually easy" sounds (instruments, animals) and perceptually rich environmental textures (wind, rain, fire, water).

## Method

All four encoders are **frozen**: each clip is passed through Audio-MAE (768-d), DINOv2 (768-d), CLAP (512-d), and CLIP (512-d) exactly once, and the embeddings are cached to disk (`02_feature_extraction.ipynb`). Training then operates only on cached embeddings, which makes every bridge cheap to train (minutes on a single T4) and guarantees all methods see identical inputs.

Bridges are trained with **InfoNCE** (temperature 0.07, batch size 512 → 511 in-batch negatives) to map audio embeddings into the corresponding visual space; outputs are L2-normalized so training and cosine-similarity retrieval share the same unit-sphere geometry. All bridges use the same optimizer (AdamW, lr 1e-4, weight decay 0.01), cosine LR annealing, gradient clipping, and early stopping on validation loss. Because cached-feature "epochs" are only 2 gradient steps each, budgets are scaled accordingly (up to 1,000 epochs, patience 100).

Architectures compared, each isolating one question:

- **Linear probe** (768→768): is a full-rank linear map sufficient? (no bottleneck, no nonlinearity)
- **ReLU MLP** (768→128→128→768): same capacity and bottleneck as SpectralBridge with standard activations — isolates the contribution of the Fourier+SIREN block
- **SpectralBridge** (768 → frozen random Fourier features (64-d, σ=10) → 2 SIREN layers (128-d) → 768): the proposed frequency-aware bridge
- **CLAP→CLIP trained MLP** (512→128→512): the fairest version of the text-mediated pipeline

## Evaluation protocol

All metrics are computed **strictly on the held-out validation split**: queries are the 254 val audio embeddings, the gallery contains only the 254 val images. Recall@K asks whether the correct image ranks in the top K by cosine similarity.

Two issues in earlier versions of this experiment were identified and corrected:

- **Evaluation leakage.** An earlier evaluation used all 1,269 pairs as both queries and gallery, letting training samples into the gallery and inflating scores for trained (memorizing) models.
- **Under-training.** With 2 gradient steps per epoch, an earlier budget (100 epochs, patience 15) stopped most bridges after ~30–40 steps. The CLAP→CLIP baseline in particular appeared 7.6× weaker than the linear probe largely because it had not finished learning; under a proper budget the gap shrinks to ~1.25×.

Both corrections changed the headline numbers substantially, which is why they are documented here rather than hidden.

## Why does the proposed bridge fail?

The ablation localizes the failure to the Fourier+SIREN block; the mechanism is hypothesized, not yet tested:

- The frozen random 768→64 projection discards most of the embedding's information irreversibly; the ReLU MLP's *learned* 768→128 compression can choose what to keep, the random one cannot.
- Fourier feature mappings and SIREN (including σ=10, w0=30, and the SIREN init scheme) were designed for **low-dimensional coordinate inputs** (e.g., pixel positions in [-1,1]). Applied to 768-d embedding inputs, the high-frequency sinusoids may scramble rather than enrich the input.
- More broadly: the periodicity that motivated the design lives in the **raw signals**, not necessarily in their embeddings — by the time Audio-MAE has summarized a waveform into 768 features, the periodic structure has already been abstracted away.

## Discussion

The linear-alignment result connects to a broader question about representation convergence (cf. the Platonic Representation Hypothesis): two encoders that never saw each other's modality appear to organize the world along largely compatible axes, differing mainly by a coordinate transform.

The current experiment, however, **cannot distinguish** category-level alignment ("wind audio maps to the wind-image region") from instance-level alignment ("this wind clip maps to *its own* frame"). With 16 categories and ~16 val examples per category, a bridge that only learned categories could already score well at R@5. Distinguishing the two readings requires **within-category retrieval** (gallery restricted to a single category) or probing embeddings for continuous physical factors (e.g., wind intensity) — both left as future work, along with scaling beyond 1,269 pairs.

## Repository structure

```
notebooks/
  01_download_vggsound.ipynb           # dataset download (yt-dlp, VGGSound subset)
  02_feature_extraction.ipynb          # frozen encoders → cached .npy embeddings
  03_train_and_evaluate_bridges.ipynb  # bridges, baselines, evaluation (this experiment)
```

## Reproducibility

All training runs in `03_train_and_evaluate_bridges.ipynb` are seeded (`torch.manual_seed(42)`, `np.random.seed(42)`) and the full notebook runs end-to-end on a single Colab T4 in well under an hour, given the cached embeddings. The train/val split uses a fixed `RandomState(42)` permutation shared by all methods. Note that exact retrieval numbers may vary by a small margin across runs due to GPU non-determinism; reported numbers are from a single clean end-to-end run, matching the committed notebook outputs.
