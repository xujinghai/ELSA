# ELSA

ELSA is a semantic segmentation project built on Detectron2 and MaskFormer for aerial urban scene understanding. This open-source release focuses on three datasets:

- `UAVid`
- `UDD6`
- `AeroScapes`

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
|  |  `- Labels_processed/
|- UDD6/
|  |- train/
|  |  |- src/
|  |  |- gt/
|  |  `- gt_idx/
|  |- val/
|  |  |- src/
|  |  |- gt/
|  |  `- gt_idx/
|- Aeroscapes/
|  |- train/
|  |  |- JPEGImages/
|  |  |- SegmentationClass/
|  |- val/
|  |  |- JPEGImages/
|  |  |- SegmentationClass/
```

## Label Preprocessing

`UAVid` uses RGB semantic labels and should be converted to class-index masks:

```bash
python datasets/UAVid/preprocess_uavid.py --dataset_path datasets/UAVid
```

`UDD6` also requires label conversion before training:

```bash
python datasets/UDD6/convert_labels.py --dataset_path datasets/UDD6
```

`AeroScapes` masks are already class-index PNGs (0-10, 255=unlabeled), so no color-to-index conversion is needed. The preprocessing script verifies the mask format and splits the official flat layout (`JPEGImages/` + `SegmentationClass/` + `ImageSets/trn.txt` + `val.txt`) into the `train/` and `val/` directories:

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

- `configs/uavid/semantic-segmentation/Base-UAVid-SemanticSegmentation.yaml` (`./output_uavid`)
- `configs/udd6/semantic-segmentation/Base-UDD6-SemanticSegmentation.yaml` (`./output_UDD`)
- `configs/Aeroscapes/semantic-segmentation/Base-Aeroscapes-SemanticSegmentation.yaml` (`./output_aeroscapes`)

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
- [PBSeg](https://github.com/zhangda1018/PBSeg)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former)
