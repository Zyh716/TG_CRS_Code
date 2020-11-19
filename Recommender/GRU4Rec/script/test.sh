python run2.py \
    --exp_name v11171_test \
    --gpu 3 \
    --lr_GRU 1e-4 \
    --num_layers 1 \
    --dropout_hidden 0.1 \
    --load_model \
    --load_model_path saved_model/v11171/gru_v11171.pth \
    --do_eval \
    # --raw