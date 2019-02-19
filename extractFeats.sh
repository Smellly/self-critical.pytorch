IMAGE_ROOT=data/coco_imgs

CUDA_VISIBLE_DEVICES=0 \
    python scripts/prepro_feats.py \
        --input_json data/dataset_coco.json \
        --output_dir data/coco0068_vc \
        --images_root $IMAGE_ROOT \
        --visual_concepts 1 \
        --vocab_size 4267 \
        --model 'vConcept101'

        # --att_size 7 \
        # --visual_concepts 0 \
        # --vocab_size 4267 \
        # --model 'vConcept101'
        # --model 'resnet101'
