LOGPATH=vc_0068_scene7_leg3_bu2_adam
echo $LOGPATH
CUDA_VISIBLE_DEVICES=3 \
    python train.py \
        --id $LOGPATH \
        --checkpoint_path log_$LOGPATH \
        --start_from log_$LOGPATH \
        --caption_model scenetopdown \
        --input_json data/cocotalk.json \
        --input_fc_dir data/cocotalk_fc \
        --input_att_dir data/cocotalk_att \
        --input_label_h5 data/cocotalk_label.h5 \
        --use_ln 0 \
        --use_scene 1 \
        --batch_size 100 \
        --learning_rate 5e-5 \
        --save_checkpoint_every 6000 \
        --language_eval 1 \
        --val_images_use 5000 \
        --self_critical_after 30 \
        --fc_feat_size 2048 \

echo $LOGPATH
