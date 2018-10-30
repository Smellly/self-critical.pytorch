CUDA_VISIBLE_DEVICES=1 \
    python eval.py \
        --dump_images 0 \
        --num_images 5000 \
        --verbose 0 \
        --verbose 0 \
        --model log_topdown/model.pth \
        --infos_path log_topdown/infos_topdown.pkl \
        --language_eval 1 
