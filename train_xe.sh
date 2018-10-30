CUDA_VISIBLE_DEVICES=2 \
    python train.py \
        --id mcbtopdown \
        --caption_model mcbtopdown \
        --input_json data/cocotalk.json \
        --input_fc_dir data/cocobu_fc \
        --input_att_dir data/cocobu_att \
        --input_label_h5 data/cocotalk_label.h5 \
        --batch_size 64 \
        --learning_rate 5e-4 \
        --learning_rate_decay_start 0 \
        --scheduled_sampling_start 0 \
        --checkpoint_path log_mcbtopdown \
        --save_checkpoint_every 6000 \
        --val_images_use 5000 \
        --max_epochs 15
