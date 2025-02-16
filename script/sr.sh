for model in Gemma2_9b_chat 
do
    for dataset in prontoqa proofwriter
    do
        
        python llm_reason.py --method bridge --n_samples 200 --model $model --dataset $dataset --sc_num 3 --weighted --remote
    done 
done