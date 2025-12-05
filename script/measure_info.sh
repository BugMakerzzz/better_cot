for model in Mistral_7b_chat 
do
    for method in cans qcot qans
    # for dataset in coinflip lastletter
    do
        for dataset in folio
        do
            python measure_info.py --dataset $dataset --task $method --n_samples 500 --model $model --bridge
        done
    done
done 