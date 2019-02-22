CUDA_VISIBLE_DEVICES=1 \
    python eval_scene.py \
        --dump_images 0 \
        --num_images 5000 \
        --cnn_model vConcept101 \
        --verbose 0 \
        --model log_vc_0068_scene_adam/model-best.pth \
        --infos_path log_vc_0068_scene_adam/infos_vc_0068_scene_adam-best.pkl \
        --input_scene_dir /home/smelly/data/place365/cocoplaces \
        --language_eval 1 

        # --cnn_model vConcept101 \
        # --cnn_model resnet101 \
