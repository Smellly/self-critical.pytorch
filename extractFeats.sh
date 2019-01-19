IMAGE_ROOT=data/coco_imgs

CUDA_VISIBLE_DEVICES=3 \
    python scripts/prepro_feats.py \
        --input_json data/dataset_coco.json \
        --output_dir data/coco0064 \
        --images_root $IMAGE_ROOT \
        --model 'vConcept101'
