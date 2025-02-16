llama2_7b_path = '/mnt/{dir}/huggingface/llama-2-7b-hf'
# llama2_7b_path = '/netcache/huggingface/llama-2-7b-hf'
llama2_7b_chat_path = '/mnt/{dir}/huggingface/llama-2-7b-chat-hf'
llama2_13b_path = '/mnt/{dir}/huggingface/Llama-2-13b-hf'
llama2_13b_chat_path = '/mnt/{dir}/huggingface/Llama-2-13b-chat-hf'
llama3_8b_path = '/mnt/{dir}/huggingface/Meta-Llama-3-8B'
llama3_8b_chat_path = '/mnt/{dir}/huggingface/Meta-Llama-3-8B-Instruct'
llama3_1_8b_chat_path = '/mnt/{dir}/huggingface/Meta-Llama-3.1-8B-Instruct-new'
llama_moe_path = '/mnt/{dir}/huggingface/LLaMA-MoE-v1-3_5B-4_16'

mistral_7b_path = '/mnt/{dir}/huggingface/Mistral-7B-v0.1'
mistral_7b_chat_path = '/mnt/{dir}/huggingface/Mistral-7B-Instruct-v0.2'

vicuna_7b_path = '/mnt/{dir}/huggingface/vicuna-7b'
vicuna_13b_path = '/mnt/{dir}/huggingface/vicuna-13b'

qwen1_8b_path = '/mnt/{dir}/huggingface/Qwen-1_8B'
qwen2_5_3b_chat_path =  '/mnt/{dir}/huggingface/Qwen2.5-3B-Instruct'
qwen2_5_7b_chat_path =  '/mnt/{dir}/huggingface/Qwen2.5-7B-Instruct'
qwen2_5_14b_chat_path = '/mnt/{dir}/huggingface/Qwen2.5-14B-Instruct'

phi2_path = '/mnt/{dir}/huggingface/phi-2'
phi3_path = '/mnt/{dir}/huggingface/Phi-3-small-8k-instruct'

yi_1_5_6b_chat_path = '/mnt/{dir}/huggingface/Yi-1.5-6B-Chat'

gemma_2_9b_path = '/mnt/{dir}/huggingface/gemma-2-9b'
gemma_2_9b_chat_path = '/mnt/{dir}/huggingface/gemma-2-9b-it'



# openai config
# OPENAI_API_KEY = 'fGBdIFoaDUeLQ'
# OPENAI_API_KEY = 'sk-pmQhVHo6pB9Kw2sSAc2840C427564aB7Ab416aBf6988A7E6'
# OPENAI_API_KEY = 'sk-i7t4FKCdavAisTCWFc2f9737854348F29d17C9E7De2e9d9e'
# OPENAI_API_KEY = 'sk-Hqe7pVTRou2RaLxE1dA85602Cc9a4eC8871f0a1839603fB5'
OPENAI_API_KEY = 'sk-oDPa1p1Pa21UJiv71YwTlcscbpFdREhYrXlPeUDp3JhwdRhY'

MAX_REQUESTS_PER_MINUTE = 3500 # 3_000 * 0.5
MAX_TOKENS_PER_MINUTE = 90000 #250_000 * 0.5
# REQUEST_URL = 'https://ai.liaobots.work/v1/chat/completions'
REQUEST_URL = 'https://api2.aigcbest.top/v1/chat/completions'

figure_colors = ['#90BCD5', '#E76254', '#FFD06F', '#7976A2', '#4A5E65', '#E29957', '#86B5A1', '#B95A58', '#4292C6']

datasets_with_options = ['aqua', 'csqa', 'ecqa', 'folio', 'logiqa', 'proofwriter', 'siqa', 'wino']

deberta_path = '/mnt/usercache/huggingface/deberta-v2-xxlarge-mnli'