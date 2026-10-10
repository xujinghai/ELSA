CONFIG="configs/uavid/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml"

python train_net.py \
    --eval-only \
    --config-file $CONFIG \
    --num-gpus 1 \
    MODEL.WEIGHTS output_potsdam/model_0004999.pth \
    OUTPUT_DIR analysis/eval_iter_4999 \
    2>&1 | tee analysis/eval_4999.log
