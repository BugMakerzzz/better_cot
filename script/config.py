#data_path
csqa_train_data_path = '../data/CommonsenseQA/train_rand_split.jsonl'
csqa_dev_data_path = '../data/CommonsenseQA/dev_rand_split.jsonl'

wino_train_data_path = '../data/winogrande_1.1/train_l.jsonl'
wino_dev_data_path = '../data/winogrande_1.1/dev.jsonl'

gsm8k_train_data_path = '../data/grade-school-math/grade_school_math/data/train.jsonl'
gsm8k_dev_data_path = '../data/grade-school-math/grade_school_math/data/test.jsonl'

proofwriter_dev_data_path = '../data'


OPENAI_API_KEY = 'sk-1Xqps1OjUfWXge5j70F10641Ee0a45Bb85667f030eD1589b'
max_requests_per_minute = 3500 # 3_000 * 0.5
max_tokens_per_minute = 90000 #250_000 * 0.5

# max_requests_per_minute = 60 # 3_000 * 0.5
# max_tokens_per_minute = 60000 #250_000 * 0.5
request_url = 'https://abc.gptmf.top/v1/chat/completions'
# request_url = "https://api.openai.com/v1/chat/completions"


llama2_7b_path = '/mnt/publiccache/huggingface/llama-2-7b-hf'
# llama2_7b_path = '/netcache/huggingface/llama-2-7b-hf'
llama2_7b_chat_path = '/mnt/publiccache/huggingface/llama-2-7b-chat-hf'
llama2_13b_path = '/mnt/publiccache/huggingface/Llama-2-13b-hf'
llama2_13b_chat_path = '/mnt/publiccache/huggingface/llama-2-13b-chat-hf'

mistral_7b_path = '/mnt/publiccache/huggingface/Mistral-7B-v0.1'
