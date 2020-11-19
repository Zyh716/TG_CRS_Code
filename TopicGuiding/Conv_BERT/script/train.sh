let gpu_id=$1

python run2.py \
    --exp_name v6 \
    --init_bert_from_pretrain \
    --gpu $gpu_id \
    # --raw
