python run2.py \
    --model_type SASRec \
    --exp_name SASRec_test \
    --do_eval \
    --gpu 1 \
    --load_exp_name SASRec_1 \
    --load_model \
    --max_seq_length 100 \
    --batch_size 256 \
    # --raw