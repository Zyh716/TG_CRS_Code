## TG-CRS

[Towards Topic-Guided Conversational Recommender System ](https://github.com/RUCAIBox/TG-ReDial).
Kun Zhou, Yuanhang Zhou, Wayne Xin Zhao, Xiaoke Wang and Ji-Rong Wen.
To appear in *International Conference on Computational Linguistics*, 2020

[ARXIV](https://arxiv.org/abs/2010.04125)



## Environment

python==3.8.6

torch==1.6.0



## Getting Started
### Installation

```
pip install -r requirements.txt
```



### Dataset

[TG-ReDial](https://github.com/RUCAIBox/TG-ReDial)



### How to use the code

0. Get data

   This repository only contains the code of TG-ReDial, and don't contain any data . You can get original data in this [repository](https://github.com/RUCAIBox/TG-ReDial). 

   To make the code more convenient, we have preprocessed all the data and prepared all trained model parameters, you can get them from [Google Drive](). You can download them and use our script to place them in correct location,  then you can directly test the model or retrain the model.  

1. Recommender

    1.1 Ours
    
    ```
    cd Recommender/Union
    # training
    bash script/train_Ours.sh
    # testing
    bash script/test_Ours.sh
    ```
    1.2 BERT
    
    ```
    cd Recommender/Union
    # training
    bash script/train_BERT.sh
    # testing
    bash script/test_BERT.sh
    ```
    1.3 TextCNN
    ```
    cd Recommender/TextCNN
    # training
    bash script/train.sh
    # testing
    bash script/test.sh
    ```
    1.4 SASRec
    ```
    cd Recommender/Union
    # training
    bash script/train_SASRec.sh
    # testing
    bash script/test_SASRec.sh
    ```
    1.5 GRU4Rec
    ```
    cd Recommender/GRU4Rec
    # training
    bash script/train.sh
    # testing
    bash script/test.sh
    ```
    1.6 KBRD
    ```
    cd Conversation/KBRD 
    bash scripts/both.sh <num_exps> <gpu_id>
    ```
    1.7 ReDial
    ```
    cd Conversation/KBRD 
    bash scripts/baseline.sh <num_exps> <gpu_id>
    ```
2. Response Generation

	2.1 Ours
   
    ```
    cd Conversation/Union
    
    # Prepare the predicted data, note that we have prepared, so you can skip this step
    #	To run ours Response Generation Model, we need to use movie predicted by 
    # ours recommender model and topic predicted by ours topic prediction model. 
    # After train the latter two models, you can use this command to get the 
    # predicted consequence
    bash ../../TopicGuiding/Ours/script/test.sh <gpu_id>
    cp ../../TopicGuiding/Ours/data/identity2topicId.json data/data_Ours
    bash ../../Recommender/Union/script/gen_pred_mids.sh
    cp ../../Recommender/Union/data/data_p_Ours/identity2movieId.json data/data_Ours
    
    # prepare for data, note that we have prepared, so you can skip this step
    bash script/Ours/prepare_data.sh
    
    # training
    bash script/Ours/train.sh
    # testing ppl
    bash script/Ours/test_ppl.sh
    # generating
    bash script/Ours/generate.sh
    # eval generation
    bash script/Ours/test_gene_metric.sh generation/v11051_gen_output.txt
    ```
    2.2 GPT2
    ```
    cd Conversation/Union
    # prepare for data, note we have prepared, so you can skip this step
    bash script/GPT2/prepare_data.sh
    # training
    bash script/GPT2/train.sh
    # testing ppl
    bash script/GPT2/test_ppl.sh
    # generating
    bash script/GPT2/generate.sh
    # eval generation
    bash script/GPT2/test_gene_metric.sh generation/v1116_gpt2_gen_output.txt
    ```
    2.3 Transformer
    ```
    cd Recommender/Transformer
    # training
    bash script/Transformer/train.sh
    # testing ppl and generating
    bash script/Transformer/test_ppl.sh
	# eval generation
	bash script/test_gene_metric.sh output/output_test_both_epoch_-1.txt
	 ```
	 2.4 KBRD
	 ```
    cd Recommender/KBRD
    # training and testing ppl
    bash scripts/t2t_rec_rgcn.sh <num_exps> <gpu_id>
    # generating and eval generation
    bash myscript/generate.sh 
	 ```
3. Topic prediction

   ```
   cd TopicGuiding/Model_You_Want
   # training
   bash script/train.sh
   # testing
   bash script/test.sh
   ```



## Reference

If you use our code, please kindly cite our papers. [Towards Topic-Guided Conversational Recommender System](https://arxiv.org/abs/2010.04125)

```
@inproceedings{zhou2020topicguided,
  title={Towards Topic-Guided Conversational Recommender System}, 
  author={Kun Zhou and Yuanhang Zhou and Wayne Xin Zhao and Xiaoke Wang and Ji-Rong Wen},
  booktitle = {Proceedings of the 28th International Conference on Computational
               Linguistics, {COLING} 2020, Barcelona, Spain, December 8-11,
               2020},
  year      = {2020}
}
```