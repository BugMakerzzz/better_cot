from load_data import DataLoader
from config import max_requests_per_minute, max_tokens_per_minute, OPENAI_API_KEY, request_url
import re
import argparse  # for running script from command line
import asyncio  # for running API calls concurrently
import aiohttp
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field  # for storing API inputs, outputs, and metadata
from transformers import set_seed
import os

async def process_api_requests(
    request_ls: list,
    results: list,
    request_url: str,
    api_key: str,
    model: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name = 'cl100k_base',
    temperature = 1.0,
    sample_cnt = 1,
    max_attempts = 100,
    logging_level = 20,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = 'chat/completions'
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    list_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug(f"Initialization complete.")

    # initialize file reading
        # `requests` will provide requests one at a time
    requests = request_ls.__iter__()

    while True:
        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
            elif list_not_finished:
                try:
                    # get new request
                    request_json = {"model":model, "seed":17, "max_tokens":200, "messages":next(requests), "temperature":temperature, "n":sample_cnt}
                    next_request = APIRequest(
                        task_id=next(task_id_generator),
                        request_json=request_json,
                        token_consumption=num_tokens_consumed_from_request(request_json, api_endpoint, token_encoding_name),
                        attempts_left=max_attempts,
                        metadata=request_json.pop("metadata", None)
                    )
                    status_tracker.num_tasks_started += 1
                    status_tracker.num_tasks_in_progress += 1
                    logging.debug(f"Reading request {next_request.task_id}: {next_request}")
                except StopIteration:
                    # if file runs out, set flag to stop reading it
                    logging.debug("Read file exhausted")
                    file_not_finished = False

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        results=results,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        status_tracker=status_tracker,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logging.warn(f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}")

# dataclasses

@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        status_tracker: StatusTracker,
        results: list
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        # logging.debug(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json,
                    proxy='http://Sept:20001228@127.0.0.1:14396'
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            results.append(data)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1

# functions

def api_endpoint_from_url(request_url):
    """Extract the API endpoint from the request URL."""
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""
    encoding = tiktoken.get_encoding(token_encoding_name)
    # if completions request, tokens = prompt + n * max_tokens
    if api_endpoint.endswith("completions"):
        max_tokens = request_json.get("max_tokens", 15)
        n = request_json.get("n", 1)
        completion_tokens = n * max_tokens
        
        # chat completions
        if api_endpoint.startswith("chat/"):
            num_tokens = 0
            for message in request_json["messages"]:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens -= 1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens + completion_tokens
        # normal completions
        else:
            prompt = request_json["prompt"]
            if isinstance(prompt, str):  # single prompt
                prompt_tokens = len(encoding.encode(prompt))
                num_tokens = prompt_tokens + completion_tokens
                return num_tokens
            elif isinstance(prompt, list):  # multiple prompts
                prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
                num_tokens = prompt_tokens + completion_tokens * len(prompt)
                return num_tokens
            else:
                raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
    # if embeddings request, tokens = input tokens
    elif api_endpoint == "embeddings":
        input = request_json["input"]
        if isinstance(input, str):  # single input
            num_tokens = len(encoding.encode(input))
            return num_tokens
        elif isinstance(input, list):  # multiple inputs
            num_tokens = sum([len(encoding.encode(i)) for i in input])
            return num_tokens
        else:
            raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
    # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
    else:
        raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1 


def chat_generate(model, input_ls, method):
    if method == 'cot_sc':
        sample_cnt = 5
        temperature = 1.0
    else:
        sample_cnt = 1
        temperature = 0.7
    results = []
    if model == 'gpt3.5':
    # openai_model = 'gpt-3.5-turbo-0613'
        openai_model = 'gpt-3.5-turbo-1106'
    else:
        openai_model = 'gpt-4-1106-preview'
    asyncio.run(
        process_api_requests(
            request_ls = input_ls,
            request_url = request_url,
            api_key=OPENAI_API_KEY,
            model=openai_model,
            max_requests_per_minute=float(max_requests_per_minute),
            max_tokens_per_minute=float(max_tokens_per_minute),
            results=results,
            temperature=temperature,
            sample_cnt=sample_cnt
        ))
    msg_dic = {}
    for inputs in input_ls:
        question = inputs[-1]['content']
        msg_dic[question] = None
    for result in results:
        question = result[0]['messages'][-1]['content']
        if sample_cnt == 1:
            response = result[1]['choices'][0]['message']['content']
            answer = extract_answer(dataset, response)
        else:
            responses = []
            answers = []
            for i in range(sample_cnt):
                response = result[1]['choices'][i]['message']['content']
                answer = extract_answer(dataset, response)
                responses.append(response)
                answers.append(answer)
            response = responses
            answer = max(answers, key=answer.count)
        msg_dic[question] = {'response':response, 'answer':answer}
    return msg_dic


def extract_answer(dataset, output):
    def extract_logic(answer):
        pattern1 = r"correct \w+ is:?\s*([A-D])"
        pattern2 = r"correct option is: (true|false|unknown)"
        pattern3 = r"([A-C])\)\s*(True|False|Unknown)"
        pattern4 = r"([A-D])\) "
        pattern5 = r"^[A-D]\.?$"

        match = re.search(pattern1, answer)
        option = None
        # extract pattern
        if match:
            option = match.group(1)
        
        if not option:
            match = re.search(pattern2, answer, re.IGNORECASE)
            if match:
                word_to_option = {"true": "A", "false": "B", "unknown": "C"}
                option = word_to_option.get(match.group(1).lower())

        if not option:
            match = re.search(pattern3, answer, re.IGNORECASE)
            if match:
                option = match.group(1)
        if not option and len(answer)<16:
            if 'true' in answer.lower():
                option = 'A'
            elif 'false' in answer.lower():
                option = 'B'
            elif 'unknown' in answer.lower():
                option = 'C'
        if not option:
            match = re.match(pattern4, answer)
            if match:
                option = match.group(1)
        if not option:
            match = re.match(pattern5, answer)
            if match:
                option = match.group(0) 
        if not option:
            option = None
            # wrong_data.append(d)
        return option
    
    if dataset in ['gsm8k', 'addition', 'product', 'gsmic']:
        answer = output.replace(',', '')  # remove middle ',' from numbers like '1,234'
        answer = re.findall('\d+', answer)
        if len(answer) == 0:
            return 'None'
        answer = answer[-1]
        answer = answer.strip()
        return str(int(answer))  # expect integer only
    elif dataset == 'proofwriter':
        answer = extract_logic(output)
        return str(answer)
    elif dataset == 'logiqa':
        answer = extract_logic(output)
        return str(answer)
    elif dataset == 'folio':
        answer = extract_logic(output)
        return str(answer)
    else:
        return output 

def prepare_inst_input(input):
    inst_inputs = []
    examples = input.split('####')
    inst_inputs.append({"role":"system", "content": examples[0]})
    for example in examples[1:-1]:
        inp, out = example.split('# Reasoning:\n')
        inp = inp + '# Reasoning:\n'
        inst_inputs.append({"role":"user", "content": inp})
        inst_inputs.append({"role":"assistant", "content": out})
    inst_inputs.append({"role":"user", "content": examples[-1]})
    return inst_inputs        
        
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gpt3.5')
parser.add_argument('--n_samples', type=int, default=500)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--dataset', type=str, default='proofwriter')
parser.add_argument('--method', type=str, default='cot')
args = parser.parse_args()

model_name = args.model
n_samples = args.n_samples
n_examples = args.n_examples
dataset = args.dataset 
method = args.method

set_seed(17)
dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
data = dataloader.load_data(method=method, n_examples=n_examples)
inst_inputs = []
for item in data:
    input = item['question']
    inst_inputs.append(prepare_inst_input(input))
    
responses = chat_generate(model_name, inst_inputs, method)
responses = list(responses .values())

i = 0
correct = 0
result = []
for item in data:
    if 'reason' in item.keys():
        reason = item['reason']
    else:
        reason = None 
    response = responses[i]['response']
    answer = responses[i]['answer']
    cor_flag = False
    if answer == item['answer']:
        cor_flag = True
        correct += 1
    msg = {'id':item['id'], 'question':item['raw_question'], 'response':response, 'answer':answer, 'reason':reason, 'label':item['answer'], 'cor_flag':cor_flag}
    result.append(msg)
    i += 1   
result.append({'acc': correct / n_samples})

result_dir = f'../result/{dataset}/{model_name}/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_path = os.path.join(result_dir, f'{method}_e{n_examples}_s{n_samples}.json')
with open(result_path, 'w') as f:
    json.dump(result, f, indent=4)
    f.close()