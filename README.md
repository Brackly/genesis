# genesis
# 🧬 GENESIS - GENerative Engine System for Image Synthesis

## Overview

GENESIS is a modular framework for exploring generative models in the image domain. 

## Research Focus

Investigating how neural networks learn to encode and reconstruct images through various architectural approaches:

- **Generative Reconstruction**: Exploring encoding/decoding architectures with different bottleneck constraints
- **Latent Space Structure**: Understanding how different models organize information in latent representations
- **Conditional Generation**: Experimenting with class-conditional generation and controllable synthesis
- **Corruption & Restoration**: Learning reversible transformations and structured degradation patterns

## Current Questions

- What's the minimum latent dimensionality needed for faithful reconstruction?
- How do different conditioning strategies affect generation quality?
- Can models learn to both apply and reverse artistic corruptions?
- How does latent space organization differ across architectures?
- What happens when we mix encoders and decoders from different trained models?

## Project Structure
```
genesis/
├── configs/          # Experiment configurations
├── src/
│   ├── models/       # Model architectures
│   ├── dataset/         # Dataset loaders
│   ├── training_engine/     # Training pipelines
│   ├── factories/    # Object creation from configs
│   └── utils/        # Visualization, metrics
├── scripts/          # Training and evaluation scripts
└── experiments/      # Experiment outputs and logs
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```python
# Train a model
python scripts/train.py --config configs/model/conditional_autoencoder.yaml

# Evaluate
python scripts/evaluate.py --checkpoint experiments/exp_001/model.pt
```

## Current Experiments

- Cross-species latent interpolation on 10-class animal dataset
- Progressive corruption networks with controllable degradation levels
- Bottleneck capacity analysis across different compression ratios
- Class embedding strategies for conditional decoders

## Architecture Philosophy

Clean separation between model definitions, data pipelines, and training logic. Models don't know about data loaders, trainers don't know about model internals. This allows rapid prototyping without refactoring existing code.

## Dependencies

- PyTorch

---
