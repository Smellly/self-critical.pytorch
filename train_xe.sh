CUDA_VISIBLE_DEVICES=0 \
    python train.py \
        --id vConcept_1_noln \
        --checkpoint_path log_vConcept_1_noln \
        --start_from log_vConcept_1_noln \
        --cnn_model vConcept \
        --caption_model topdown \
        --use_ln 0 \
        --input_json data/cocotalk.json \
        --input_label_h5 data/cocotalk_label.h5 \
        --input_fc_dir data/cocomytalk_fc \
        --input_att_dir data/cocomytalk_att \
        --losses_log_every 10 \
        --batch_size 128  \
        --learning_rate 5e-4 \
        --learning_rate_decay_start 0 \
        --scheduled_sampling_start 0 \
        --save_checkpoint_every 1000 \
        --val_images_use 5000 \
        --max_epochs 20

        
        # --input_fc_dir data/cocobu_fc \
        # --input_att_dir data/cocobu_att \
