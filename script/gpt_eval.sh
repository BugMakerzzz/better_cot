for dataset in aqua ecqa gsm8k lastletter prontoqa proofwriter
do
    python gpt_eval.py --model $1 --dataset $dataset
done