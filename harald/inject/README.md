# Qwen3-VL to Qwen-Image Linear Injection POC

Minimal proof-of-concept for the textual inversion-inspired approach to inject person identity/state into Qwen-Image generation via Qwen3-VL visual features.

## Overview

This POC demonstrates:
1. Extracting visual features from person photos using Qwen3-VL
2. Training a linear head to map features → Qwen-Image prompt embeddings (residual)
3. Generating images with identity injection at different strengths (alpha sweep)

## Architecture

```
Photo (448x448) → Qwen3-VL → Features [D]
                                ↓
                         Linear Head (+ scale)
                                ↓
                         Residual [1, S, H]
                                ↓
Base Prompt Embeds + α × Residual → Qwen-Image → Generated Image
```

## Usage

### On modal.com

The POC runs on modal.com as configured in `harald/modal_runner.py`:

```bash
# Full workflow (training + inference)
modal run -m harald.modal_runner::injection_poc \
  --seed-dirs /mnt/dataset/8/seed_2816178403,/mnt/dataset/8/seed_1461687847,/mnt/dataset/8/seed_2732585231 \
  --train \
  --epochs 5

# Inference only (requires pre-trained checkpoint)
modal run -m harald.modal_runner::injection_poc \
  --seed-dirs /mnt/dataset/8/seed_2816178403,/mnt/dataset/8/seed_1461687847

# Custom parameters
modal run -m harald.modal_runner::injection_poc \
  --seed-dirs /mnt/dataset/8/seed_2816178403,/mnt/dataset/8/seed_1461687847 \
  --train \
  --epochs 3 \
  --batch-size 2 \
  --lr 1e-3 \
  --alphas "0.5,1.0,1.5,2.0" \
  --gen-seeds "1234,5678,9999" \
  --base-prompt "a portrait photo of a person, studio lighting, 85mm"

# Minimal example (using defaults)
modal run -m harald.modal_runner::injection_poc \
  --seed-dirs /mnt/dataset/8/seed_2816178403,/mnt/dataset/8/seed_1461687847 \
  --train
```

**Important Notes:**
- **No `--` separator**: Modal arguments are passed directly (no `--` between command and arguments)
- **Comma-separated values**: `alphas` and `gen_seeds` must be comma-separated strings (e.g., `"0.5,1.0,1.5"`)
- **seed_dirs is required**: Must provide at least one seed directory path

### Local Testing (if needed)

```bash
python -m harald.inject.poc \
  --seed-dirs /path/to/seed_123,/path/to/seed_456 \
  --out-dir ./outputs \
  --train --epochs 3
```

## Arguments

### Modal Function Parameters

When calling via `modal run -m harald.modal_runner::injection_poc`, the following parameters are available:

#### Required
- `--seed-dirs` (str): Comma-separated list of `seed_*` directory paths containing `photo_view_*.png` files

#### Training
- `--train` (flag): Enable training mode (default: False)
- `--epochs` (int): Number of training epochs (default: 3)
- `--batch-size` (int): Training batch size (default: 2)
- `--lr` (float): Learning rate (default: 1e-3)

#### Inference
- `--alphas` (str): Comma-separated alpha values for residual scaling (default: "0.5,1.0,1.5")
- `--gen-seeds` (str): Comma-separated random seeds for generation (default: "1234,5678,9999")
- `--steps` (int): Number of diffusion steps (default: 28)
- `--guidance` (float): Guidance scale (default: 4.5)
- `--height` (int): Generated image height (default: 768)
- `--width` (int): Generated image width (default: 512)

#### Prompts
- `--base-prompt` (str): Base prompt for generation (default: "a portrait photo of a person, studio lighting, 85mm")
- `--negative-prompt` (str): Negative prompt (default: "blurry, low quality, deformed, watermark")

### Direct Script Arguments (Local Use Only)

When running `inject/poc.py` directly (not via modal), these additional arguments are available:

#### Required
- `--seed-dirs`: Comma-separated list of `seed_*` directory paths containing `photo_view_*.png` files

### Optional - Data
- `--out-dir`: Output directory (default: `/mnt/output/injection_poc`)

### Optional - Models
- `--qwen3-vl-model`: Qwen3-VL model ID (default: `Qwen/Qwen2-VL-7B-Instruct`)
- `--qwen-image-model`: Qwen-Image model ID (default: `Qwen/Qwen2.5-7B-Instruct`)

### Optional - Training
- `--train`: Enable training mode
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 2)
- `--lr`: Learning rate (default: 1e-3)

### Optional - Inference
- `--alphas`: Alpha values for residual scaling (default: 0.5 1.0 1.5)
- `--gen-seeds`: Random seeds for generation (default: 1234 5678 9999)
- `--steps`: Number of diffusion steps (default: 28)
- `--guidance`: Guidance scale (default: 4.5)
- `--height`: Generated image height (default: 768)
- `--width`: Generated image width (default: 512)

### Optional - Prompts
- `--base-prompt`: Base prompt for generation (default: "a portrait photo of a person, studio lighting, 85mm")
- `--negative-prompt`: Negative prompt (default: "blurry, low quality, deformed, watermark")

### Optional - System
- `--device`: Device (default: cuda if available)
- `--dtype`: Data type - fp16/fp32/bf16 (default: fp16)
- `--hf-token`: HuggingFace token (reads from HUGGINGFACE_TOKEN env var if not provided)
- `--cache-dir`: HF cache directory (reads from HF_HUB_CACHE env var if not provided)

## Output Structure

```
/mnt/output/injection_poc/
├── checkpoints/
│   └── linear_head.pt              # Trained linear head weights
└── generations/
    ├── seed_123_alpha_grid.png     # Grid of generated images
    ├── seed_123_meta.json          # Generation metadata
    ├── seed_456_alpha_grid.png
    └── seed_456_meta.json
```

### Grid Layout

Each grid PNG contains generated images organized as:
- **Rows**: Different alpha values (residual injection strength)
- **Columns**: Different random seeds (generation diversity)

Example for `--alphas 0.5 1.0 1.5` and `--gen-seeds 1234 5678 9999`:
```
α=0.5: [seed 1234] [seed 5678] [seed 9999]
α=1.0: [seed 1234] [seed 5678] [seed 9999]
α=1.5: [seed 1234] [seed 5678] [seed 9999]
```

## Implementation Details

### Simplifications from Original Code

1. **Single image per identity**: Only uses the first `photo_view_*.png` file per seed directory
2. **Fixed resize**: Simple 448x448 center crop (no pixel budget calculations)
3. **No vision hook complexity**: Uses standard `AutoProcessor` and model forward pass
4. **Minimal training target**: Trains to predict zero residuals (identity mapping) - full version would use caption embeddings
5. **Small scale**: Designed for 5-10 identities, 3-5 epochs

### Key Components

- **Qwen3VLFeatureExtractor**: Extracts L2-normalized visual features from images
- **QwenImageTextHelper**: Encodes text to embeddings and generates images from embeddings
- **LinearResidualHead**: Maps visual features → residual embeddings with learnable scale
- **SinglePhotoDataset**: Loads single photo per identity from configured directories
- **train_linear_head()**: Training loop with MSE + cosine loss
- **run_inference()**: Alpha sweep inference with grid visualization

### Determinism

All random operations are seeded (seed=123) for reproducibility:
- Python random
- NumPy random
- PyTorch CPU/CUDA random
- cuDNN deterministic mode

## Next Steps

After validating the POC:

1. **Scale to full dataset**: Use all photo views per identity
2. **Implement caption-based training**: Use actual caption embeddings as targets
3. **Add pixel budget handling**: Multi-image processing with memory management
4. **Robust vision hooks**: Fallback paths for different Qwen3-VL versions
5. **Hyperparameter tuning**: Learning rate, loss weights, alpha ranges
6. **Evaluation metrics**: Identity preservation, image quality, diversity

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (try 1)
- Use `--dtype bf16` for better memory efficiency
- Reduce image count per identity

### Model Loading Issues
- Ensure `HUGGINGFACE_TOKEN` is set in modal secrets
- Check `HF_CACHE` volume is mounted correctly
- Verify model IDs are correct and accessible

### No Images Generated
- Check `--seed-dirs` paths are correct
- Verify `photo_view_*.png` files exist in seed directories
- Ensure checkpoint was saved during training

## Requirements

From `requirements.txt` and `harald/config.py`:
- torch >= 2.7.1+cu128
- transformers >= 4.53.0
- diffusers (git main)
- accelerate >= 0.35.0
- Pillow
- tqdm
- numpy
