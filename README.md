# ELSA

**ELSA** (**E**fficient **L**inear-attention with **S**imAM **A**rchitecture) is an efficient semantic segmentation framework for UAV (Unmanned Aerial Vehicle) imagery, built upon the mask classification paradigm of Mask2Former with a ConvNeXt-Tiny backbone.

## Architecture

ELSA consists of three main components: a **ConvNeXt-Tiny backbone** that extracts multi-scale features, a **SimAM-enhanced pixel decoder** that fuses cross-scale features with parameter-free attention, and a **Transformer decoder with Linear Cross-Attention (LCA)**  that performs query-based mask classification. The three components work complementarily: LCA reduces decoder cost, SimAM provides zero-parameter feature modulation, and ConvNeXt-Tiny serves as an accuracy-efficiency base.

## Key Features

* **Linear Cross-Attention (LCA)**: Reduces decoder complexity from quadratic to linear with query-aware bias compensation, preserving per-query discriminability while significantly cutting computational cost.

* **SimAM-Enhanced Pixel Decoder**: Integrates a parameter-free SimAM (Simple, Parameter-Free Attention Module) for feature modulation, adding zero learnable parameters while enhancing spatial discriminability.

* **ConvNeXt-Tiny Backbone**: A modernized CNN architecture that bridges CNNs and Transformers, offering a superior accuracy-efficiency trade-off for UAV deployment.

* **Mask Classification Paradigm**: Unified framework supporting semantic, instance, and panoptic segmentation via query-based mask prediction.

## Performance

This open-source release focuses on three datasets:

* `UAVid`

* `UDD6`

* `AeroScapes`

The following table provides a partial description of the datasets：

| Dataset    | Classes | Train / Val / Test |
| :--------- | :-----: | :----------------: |
| UAVid      |    8    |   200 / 70 / 150   |
| UDD6       |    6    |    90 / 16 / 35    |
| AeroScapes |    11   |  2288 / 654 / 327  |

The repository includes dataset registration, label preprocessing, training, evaluation, visualization, and basic model analysis scripts.

## Environment

The project has been tested with `Python 3.10`, `PyTorch 1.12.0`, and `torchvision 0.13.0`.

```bash
conda create -n elsa python=3.10 -y
conda activate elsa

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"
pip install -r requirements.txt
```

## Dataset Layout

Download the datasets from their official sources:

* **UAVid**: [https://uavid.nl/](https://uavid.nl/)

* **UDD6**: [https://github.com/MarcWong/UDD](https://github.com/MarcWong/UDD)

* **AeroScapes**: [https://github.com/ishann/aeroscapes](https://github.com/ishann/aeroscapes)

Place the datasets under `datasets/` with the following structure:

```text
datasets/
|- UAVid/
|  |- uavid_train/
|  |  |- Images/
|  |  |- Labels/
|  |  `- Labels_processed/
|  |- uavid_val/
|  |  |- Images/
|  |  |- Labels/
|  |   `- Labels_processed/
|- UDD6/
|  |- train/
|  |  |- src/
|  |  |- gt/
|  |  `- gt_idx/
|  |- val/
|  |  |- src/
|  |   |- gt/
|  |  `- gt_idx/
|- Aeroscapes/
|- train/
|  |- JPEGImages/
|  |   |- SegmentationClass/
|- val/
|  |- JPEGImages/
|  |  |- SegmentationClass/
```

## Label Preprocessing

### UAVid

UAVid contains 8 semantic classes: **building, road, tree, low vegetation, moving car, static car, human, and background clutter**. The original labels are RGB color-coded and must be converted to class-index masks for training.

The preprocessing script maps each RGB color tuple to its corresponding class index (0-7) and saves the result as single-channel PNG files in `Labels_processed/`. All training images are cropped to 1024×1024 during training.

```bash
python datasets/UAVid/preprocess_uavid.py --dataset_path datasets/UAVid
```

### UDD6

UDD6 (UAV Drone Dataset 6) consists of 6 semantic classes: **facade, road, vegetation, vehicle, roof, and other**. The original ground-truth labels require conversion to class-index format before training.

The conversion script transforms the original label format to indexed PNG files stored in `gt_idx/`. All images are cropped to 1024×1024 during training.

```bash
python datasets/UDD6/convert_labels.py --dataset_path datasets/UDD6
```

### AeroScapes

AeroScapes comprises 11 semantic classes: **person, bike, car, drone, boat, animal, obstacle, construction, vegetation, road, and sky**. The masks are already provided as class-index PNGs (values 0-10, with 255 representing unlabeled/ignore regions), so no color-to-index conversion is needed.

The preprocessing script verifies the mask format and reorganizes the official flat layout into `train/` and `val/` directories. All images are cropped to 1024×1024 during training.

```bash
python datasets/Aeroscapes/preprocess_aeroscapes.py --dataset_path datasets/Aeroscapes
```

## Training

Train on `UAVid`:

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/uavid/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml
```

Train on `UDD6`:

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/udd6/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml
```

Train on `AeroScapes`:

```bash
python train_net.py --num-gpus 1 \
  --config-file configs/Aeroscapes/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml
```

The default output directories are defined in the base config files:

* `configs/uavid/semantic-segmentation/Base-UAVid-SemanticSegmentation.yaml` (`./output_uavid`)

* `configs/udd6/semantic-segmentation/Base-UDD6-SemanticSegmentation.yaml` (`./output_UDD`)

* `configs/Aeroscapes/semantic-segmentation/Base-Aeroscapes-SemanticSegmentation.yaml` (`./output_aeroscapes`)

Pretrained_model:

https://drive.google.com/file/d/1fvQVFRUe36b2_kbN-QRcxqDC8W3FSjuN/view?usp=drive_link

## Evaluation

Evaluate a trained checkpoint on `UAVid`:

```bash
python train_net.py --eval-only \
  --num-gpus 1 \
  --config-file configs/uavid/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml \
  MODEL.WEIGHTS /path/to/uavid_model.pth
```

Evaluate a trained checkpoint on `UDD6`:

```bash
python train_net.py --eval-only \
  --num-gpus 1 \
  --config-file configs/udd6/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml \
  MODEL.WEIGHTS /path/to/udd6_model.pth
```

Evaluate a trained checkpoint on `AeroScapes`:

```bash
python train_net.py --eval-only \
  --num-gpus 1 \
  --config-file configs/Aeroscapes/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml \
  MODEL.WEIGHTS /path/to/aeroscapes_model.pth
```

## Visualization

Visualize `UAVid` predictions:

```bash
python visualize_uavid_predictions.py \
  --config configs/uavid/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml \
  --weights /path/to/uavid_model.pth \
  --input datasets/UAVid/uavid_val/Images \
  --gt datasets/UAVid/uavid_val/Labels \
  --output visualizations/uavid
```

Visualize `UDD6` predictions:

```bash
python visualize_udd6_predictions.py \
  --config configs/udd6/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml \
  --weights /path/to/udd6_model.pth \
  --input datasets/UDD6/val/src \
  --gt datasets/UDD6/val/gt \
  --output visualizations/udd6
```

Visualize `AeroScapes` predictions:

```bash
python visualize_aeroscapes_predictions.py \
  --config configs/Aeroscapes/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml \
  --weights /path/to/aeroscapes_model.pth \
  --input datasets/Aeroscapes/val/JPEGImages \
  --gt datasets/Aeroscapes/val/SegmentationClass \
  --output visualizations/aeroscapes
```

## Acknowledgements

ELSA builds on top of:

* [PBSeg](https://github.com/zhangda1018/PBSeg)

* [Detectron2](https://github.com/facebookresearch/detectron2)

* [Mask2Former](https://github.com/facebookresearch/Mask2Former

