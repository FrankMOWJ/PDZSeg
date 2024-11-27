python train_seg.py \
    --num_classes 2\
    --backbone_size base \
    --decoder_type segformer \
    --prompt_type None \
    --batch_size 8 \
    --epoch 100 \
    --init_lr 0.001 \
    --crop_size 532 \
    --log_dir ./log \
    --seed 3407 \
    --device cuda:0 \

    
    