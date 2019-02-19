CUDA_VISIBLE_DEVICES=3 \
    python eval.py \
        --dump_images 0 \
        --num_images 5000 \
        --cnn_model vConcept101 \
        --verbose 0 \
        --model log_vConcept_vc2_adam/model-best.pth \
        --infos_path log_vConcept_vc2_adam/infos_vConcept_vc2_adam-best.pkl \
        --language_eval 1 

        # --cnn_model vConcept101 \
        # --cnn_model resnet101 \
