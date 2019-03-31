LOGPATH=Ensemble
echo $LOGPATH

CUDA_VISIBLE_DEVICES=0 \
    python eval_scene_ensemble.py \
        --ids vc_0068_scene7_leg3bu1ln_adam_rlrl \
              vc_0068_scene7_leg3bu1ln_adam_rlrl2 \
        --dump_images 0 \
        --num_images 40504 \
        --verbose_beam 0 \
        --verbose_loss 0 \
        --beam_size 3 \
        --language_eval 1 

echo $LOGPATH
