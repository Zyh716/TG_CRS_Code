python run2.py \
    --model_type SASRec \
    --exp_name SASRec_test \
    --do_eval \
    --gpu 2 \
    --load_exp_name SASRec_0 \
    --load_model \
    --max_seq_length 60 \
    --batch_size 256 \
    --is_save_sasrec_embed

cp saved_model/sasrec_embed.pth ../GRU4Rec/data