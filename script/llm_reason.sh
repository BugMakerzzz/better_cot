for model in Gemma2_9b_chat
do
    for dataset in ecqa proofwriter aqua gsmic folio siqa 
    do
        python llm_reason.py --method direct --sc_num 10 --n_samples 200 --model $model --dataset $dataset --remote
    done 
done