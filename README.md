# SpectralBridge: Frequency-Aware Audio-Visual Alignment Beyond CLIP

**A language-free approach to audio-visual alignment using self-supervised encoders and frequency-aware neural bridges.**

## Overview

SpectralBridge demonstrates that audio and visual signals can be aligned directly without text-mediated projection (as used in CLIP-based approaches like SonicDiffusion). By using BEATs (self-supervised audio) and DINOv2 (self-supervised vision) with a Fourier-feature-enhanced bridge network, we capture perceptual audio-visual correspondences that language cannot express.

## Architecture
```
Audio → BEATs → SpectralBridge (Fourier + SIREN) → DINOv2 Space
                                                          ↓
                                                    Image Generation
```

## Key Features

- **Language-Free Alignment**: No text-mediated projection
- **Frequency-Aware**: Fourier feature mapping + SIREN activations
- **Efficient**: Trains on cached features (~2-5M parameters)
- **Practical**: Works on T4 GPU (Google Colab/Kaggle)

## Project Structure
```
spectralbridge/
├── notebooks/          # Jupyter notebooks for each phase
├── src/               # Reusable Python modules
│   ├── models/        # Network architectures
│   ├── data/          # Dataset classes
│   └── utils/         # Helper functions
├── scripts/           # Standalone scripts
├── configs/           # Configuration files
├── data/              # Data directory (gitignored)
└── outputs/           # Experiment results (gitignored)
```

## Setup

See `docs/setup.md` for detailed installation instructions.


## License

MIT License (see LICENSE file)
