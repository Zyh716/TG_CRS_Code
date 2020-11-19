generation_file=$1

# echo $generation_file
python script/GPT2/BLEU_scorer.py $generation_file
python script/GPT2/Dist_scorer.py $generation_file