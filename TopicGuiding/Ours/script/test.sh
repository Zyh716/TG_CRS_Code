let gpu_id=$1

python run2.py \
    --exp_name v1 \
    --init_from_fineturn \
    --init_add \
    --do_eval \
    --bert_path saved_model/v6_rerun \
    --gpu $gpu_id \
    # --raw

