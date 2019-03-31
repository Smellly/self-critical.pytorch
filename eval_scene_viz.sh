# LOGPATH=vc_0068_scene7_leg3ln2_share2_adam
# LOGPATH=vc_0068_scene7_leg3ln3_share_wo_adam
LOGPATH=vc_0068_scene7_leg3bu1ln_adam_rl
echo $LOGPATH

CUDA_VISIBLE_DEVICES=1 \
    python eval_scene_viz.py \
        --dump_images 0 \
        --num_images 5000 \
        --cnn_model vConcept101 \
        --verbose 1 \
        --model log_$LOGPATH/model-best.pth \
        --infos_path log_$LOGPATH/infos_$LOGPATH-best.pkl \
        --input_scene_dir /home/smelly/data/place365/cocoplaces \
        --beam_size 1 \
        --language_eval 0

        # --cnn_model vConcept101 \
        # --cnn_model resnet101 \
echo $LOGPATH
