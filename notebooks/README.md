# Segmentation Mask Visualization Notebook

Visualize predicted segmentation masks from IJEPA linear probe models using the official dataset loaders from `src/datasets`.

## Supported Datasets

### 1. **MADOS** (Multi-spectral Satellite Imagery)
- **Dataset Class**: `MADOSDataset` (HDF5-based)
- **Data Source**: `src/datasets/mados_dataset.py`
- **Input Channels**: 10 → filtered to BGR+NIR (4 channels)
- **Number of Classes**: 15
- **Image Size**: 224×224 (configurable via config)
- **Config File**: `configs/mados_linear_probe.yaml`

### 2. **M-Cashew** (Cashew Plantation Segmentation)
- **Dataset Class**: `GeobenchDataset` (GeoBench API)
- **Data Source**: `src/datasets/geobench_dataset.py`
- **Input Channels**: 4 (BGRN from Sentinel-2)
- **Number of Classes**: 7
- **Image Size**: 112×112 (configurable via config)
- **Config File**: `configs/m_cashew_linear_probe.yaml`

### 3. **M-SA-Crop-Type** (Southern Africa Crop Type)
- **Dataset Class**: `GeobenchDataset` (GeoBench API)
- **Data Source**: `src/datasets/geobench_dataset.py`
- **Input Channels**: 4 (BGRN from Sentinel-2)
- **Number of Classes**: 10
- **Image Size**: 224×224 (configurable via config)
- **Config File**: `configs/m_sa_crop_type_linear_probe.yaml`

## How to Use

### Quick Start

1. **Open the notebook**:
   ```bash
   jupyter notebook notebooks/visualize_segmentation_predictions.ipynb
   ```

2. **Select a dataset** in "1. Configuration" cell:
   ```python
   SELECTED_DATASET = 'mados'  # or 'm_cashew' or 'm_sa_croptype'
   ```

3. **Run all cells** (Kernel → Run All)

### What Each Cell Does

| # | Name | Purpose |
|---|------|---------|
| 0 | Title | Introduction |
| 1 | Imports | Load libraries and set device |
| 2 | Configuration | Load YAML config for selected dataset |
| 3 | Load Data from src/datasets | Use `make_eval_dataloaders()` to load samples |
| 4 | Load Models | Initialize pretrained encoder & linear segmentation head |
| 5 | (unnamed) | Load trained linear probe checkpoint weights |
| 6 | Inference | Run segmentation predictions on all samples |
| 7 | Visualization Utilities | Helper functions (RGB/NIR extraction, colormaps, etc.) |
| 8 | Main Visualization | 4-column view: RGB + NIR + Ground Truth + Predictions |
| 9 | Confidence & Uncertainty | Model confidence (softmax) and entropy maps |
| 10 | Evaluation Metrics | Compute and display IoU and pixel accuracy |

## Output Files

The notebook generates visualization PNG images in `notebooks/`:

- `{dataset}_predictions.png` - 4-column segmentation comparison (RGB, NIR, GT, Pred)
- `{dataset}_confidence.png` - Model confidence and uncertainty (entropy) maps

Example outputs:
```
notebooks/
├── mados_predictions.png
├── mados_confidence.png
├── m_cashew_predictions.png
├── m_cashew_confidence.png
├── m_sa_croptype_predictions.png
└── m_sa_croptype_confidence.png
```

## Configuration

The notebook uses YAML config files from the `configs/` directory. Each config contains:

```yaml
data:
  data_root: /path/to/dataset
  dataset_name: mados|m-cashew-plantation|m-SA-crop-type
  batch_size: 64
  crop_size: 224
  num_workers: 10

meta:
  model_name: vit_small
  patch_size: 14
  pretrained_checkpoint: /path/to/encoder.pth.tar
  seg_n_cls: 15           # Number of segmentation classes
  use_batchnorm: false
```

### Customizing Settings

1. **Load fewer samples for faster processing**:
   ```python
   NUM_SAMPLES = 3  # Default: 6
   ```

2. **Use different data split**:
   ```python
   SPLIT = 'test'  # 'train', 'val', or 'test'
   ```

3. **Modify config files**:
   - Edit `configs/{dataset}_linear_probe.yaml` directly
   - Change `data_root`, `crop_size`, `seg_n_cls`, etc.

## Model Loading

### Pretrained Encoder

The notebook loads pretrained Vision Transformer encoders from IJEPA pretraining:
- Path specified in config: `meta.pretrained_checkpoint`
- Positional embeddings are removed for size flexibility
- Encoder is frozen (only linear head is trainable)

### Linear Probe Head

After loading the encoder, the notebook attempts to load trained linear probe weights:

```
logs/eval/
├── mados_linear_probe_hrgram_model/
│   └── *-best.pth.tar
├── m_cashew_linear_probe_hrmodel/
│   └── *-best.pth.tar
└── m_sa_crop_type_linear_probe_hrmodel/
    └── *-best.pth.tar
```

If checkpoints aren't found, the notebook uses an untrained linear head and shows a warning.

## Data Loading Details

### Using `make_eval_dataloaders()`

The notebook uses the official dataset factory function:

```python
from src.datasets.helpers import make_eval_dataloaders

(train_loader, train_sampler,
 val_loader, val_sampler,
 test_loader, test_sampler) = make_eval_dataloaders(
    dataset_name='mados',
    data_root='/path/to/data',
    img_size=224,
    batch_size=64,
    num_workers=0,  # Set to 0 for notebooks
    world_size=1,
    rank=0
)
```

This function:
1. Detects dataset type by name
2. Routes to appropriate dataset class (MADOS or GeoBench)
3. Applies correct preprocessing (normalization, resizing, etc.)
4. Returns train/val/test loaders with samplers

### Sample Format

All samples are loaded with the same structure:

```python
{
    'image': torch.Tensor(4, H, W),     # BGRN format
    'label': torch.Tensor(H, W),        # Class indices (-1 = ignore)
    'batch_idx': int,                   # Batch index
    'sample_in_batch': int              # Position in batch
}
```

## Troubleshooting

### Error: "No checkpoint found"

The notebook will use an untrained linear probe if checkpoints aren't found. To use trained weights:

1. Check checkpoint directory exists:
   ```bash
   ls logs/eval/mados_linear_probe_hrgram_model/
   ```

2. Verify checkpoint filename contains `.pth.tar`

### Error: "geobench module not found"

For M-Cashew and M-SA-Crop-Type datasets, install GeoBench:

```bash
pip install geobench
```

### Error: "data_root not found"

Verify data paths in config files:

```bash
# Check MADOS data
ls /home/admin/AGML_ResearchGroup/SuperResolution/mados/mados.h5

# Check GeoBench data
ls /home/admin/AGML_ResearchGroup/SuperResolution/
```

### Data loading is slow

Set `num_workers=0` in the notebook (already done). Multi-worker loading can be problematic in Jupyter.

## Architecture Overview

```
Input Image (4, H, W) [BGRN]
         ↓
  [Pretrained ViT Encoder]
    frozen weights
         ↓
  Patch Embeddings (64, 384)
         ↓
  [Linear Segmentation Head]
    trained via linear probing
         ↓
  Logits (n_classes, 8, 8)
         ↓
  [Bilinear Interpolation]
         ↓
  Predictions (n_classes, H, W)
         ↓
  Class Map (H, W)
```

## Metrics Computed

- **Pixel Accuracy**: Percentage of correctly classified pixels
  - Formula: `correct_pixels / total_valid_pixels`
  - Ignores pixels with label = -1

- **IoU (Intersection over Union)**: Per-class segmentation metric
  - Formula: `intersection / union` for each class
  - Reported as Mean IoU: average across all classes

- **Entropy**: Model uncertainty per pixel
  - Higher entropy = higher uncertainty
  - Formula: `-sum(p * log(p))` for probability distribution

## Tips & Tricks

1. **Visualize specific samples**:
   - Modify `NUM_SAMPLES` to load fewer samples for faster processing
   - Change `SPLIT` to 'train', 'val', or 'test'

2. **Debug data loading**:
   - Add print statements in the data loading cell to inspect tensor shapes
   - Check that images are in BGRN format (4 channels)

3. **Analyze model confidence**:
   - Look at confidence maps to identify uncertain regions
   - High entropy areas often correspond to prediction errors

4. **Compare datasets**:
   - Run the notebook once for each dataset
   - Save visualizations and compare segmentation difficulty

## References

- IJEPA: Image Joint-Embedding Predictive Architecture (Assran et al., 2023)
- Vision Transformer (ViT): Image is Worth 16x16 Words (Dosovitskiy et al., 2021)
- GeoBench: Earth Observation Benchmark (Tuia et al.)
