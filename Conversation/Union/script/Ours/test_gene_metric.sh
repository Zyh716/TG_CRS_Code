generation_file=$1

# echo $generation_file
python script/Ours/BLEU_scorer.py $generation_file
python script/Ours/Dist_scorer.py $generation_file