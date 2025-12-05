for model in Gemma2_9b_chat
do
    for dataset in math 
    do
        python measure_faith.py --dataset $dataset --model $model --method ig 
    done 
done