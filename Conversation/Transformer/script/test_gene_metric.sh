generation_file=$1

# echo $generation_file
python script/BLEU_scorer.py $generation_file
python script/Dist_scorer.py $generation_file