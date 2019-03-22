LOGPATH=vc_0068_scene7_leg3ln2_share_adam
echo $LOGPATH

CUDA_VISIBLE_DEVICES=1 \
    python eval_scene.py \
        --dump_images 0 \
        --num_images 5000 \
        --cnn_model vConcept101 \
        --verbose 0 \
        --model log_$LOGPATH/model-best.pth \
        --infos_path log_$LOGPATH/infos_$LOGPATH-best.pkl \
        --input_scene_dir /home/smelly/data/place365/cocoplaces \
        --beam_size 2 \
        --language_eval 1 

        # --cnn_model vConcept101 \
        # --cnn_model resnet101 \
echo $LOGPATH
