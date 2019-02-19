CUDA_VISIBLE_DEVICES=1 \
    python train.py \
        --id vConcept_vc3_adam \
        --checkpoint_path log_vConcept_vc3_adam \
        --cnn_model vConcept101 \
        --caption_model topdown \
        --use_ln 0 \
        --input_json data/cocotalk.json \
        --input_label_h5 data/cocotalk_label.h5 \
        --input_fc_dir data/coco0068_vc_fc \
        --input_att_dir data/coco0068_vc_att \
        --losses_log_every 10 \
        --batch_size 100  \
        --optim adam \
        --learning_rate 5e-4 \
        --learning_rate_decay_start 0 \
        --scheduled_sampling_start 0 \
        --save_checkpoint_every 1000 \
        --val_images_use 5000 \
        --att_feat_size 4267 \
        --rnn_size 1000 \
        --input_encoding_size 1000 \
        --att_hid_size 512 \
        --max_epochs 30

        # --start_from log_vConcept_vc2_adam \
        # --learning_rate 5e-4 \
        # --input_fc_dir data/cocobu_fc \
        # --input_att_dir data/cocobu_att \
