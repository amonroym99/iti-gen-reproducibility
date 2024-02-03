
#!/bin/bash

python train_iti_gen.py \
    --prompt="a natural scene" \
    --attr-list="Colorful" \
    --epochs=30 \
    --save-ckpt-per-epochs=10 \
    --refer-size-per-category=10 \
    --ckpt-path="ckpts-10"
