import diskcache as dc
import asyncio
from tqdm import tqdm

MODEL_DICT = {
    'GPT-OSS-20B': 'openai/gpt-oss-20b',
    'GPT-OSS-120B': 'openai/gpt-oss-120b',
    'gemma3-27b': 'google/gemma-3-27b-it',
    'llama3-8b': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama3-70b': 'meta-llama/Llama-3.3-70B-Instruct',
    'Qwen3-30B': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
    'Qwen3-8B': 'Qwen/Qwen3-8B',
    'Qwen3-8B-Thinking': 'Qwen/Qwen3-8B',
    'GPT-OSS-120B-medium': 'openai/gpt-oss-120b',
    'GPT-OSS-20B-medium': 'openai/gpt-oss-20b',
    'Qwen3-Next-Thinking': 'Qwen/Qwen3-Next-80B-A3B-Thinking',
    'Qwen3-Next': 'Qwen/Qwen3-Next-80B-A3B-Instruct',
    'Qwen3-32B': 'Qwen/Qwen3-32B',
    'Qwen3-32B-Thinking': 'Qwen/Qwen3-32B',
    'gpt-5': 'gpt-5-2025-08-07',
    'gpt-5-mini': 'gpt-5-mini-2025-08-07',
    'gpt-5-nano': 'gpt-5-nano-2025-08-07',
    'gpt-4.1-mini': 'gpt-4.1-mini-2025-04-14',
    'gpt-4.1-nano': 'gpt-4.1-nano-2025-04-14',
    'deepseek-reasoner': 'deepseek-reasoner',
    'deepseek-chat': 'deepseek-chat',
    'Llama3-8B': 'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen3-14B': 'Qwen/Qwen3-14B',
    'Qwen3-14B-Thinking': 'Qwen/Qwen3-14B',
    'Gemma3-4B': 'google/gemma-3-4b',
    'Apertus-8b': 'swiss-ai/Apertus-8B-Instruct-2509',
    'Apertus-70b': 'swiss-ai/Apertus-70B-Instruct-2509',
}
INPUT_COST_DICT = {
    'GPT-OSS-20B': 0,
    'GPT-OSS-120B': 0,
    'GPT-OSS-120B-medium': 0,
    'GPT-OSS-20B-medium': 0,
    'llama3-8b': 0,
    'llama3-70b': 0,
    'gemma3-27b': 0,
    'Qwen3-30B': 0,
    'Qwen3-8B': 0,
    'Qwen3-8B-Thinking': 0,
    'Qwen3-Next': 0,
    'Qwen3-Next-Thinking': 0,
    'Qwen3-32B': 0,
    'Qwen3-32B-Thinking': 0,
    'gpt-5': 1.25,
    'gpt-5-mini': 0.25,
    'gpt-5-nano': 0.05,
    'gpt-4.1': 2,
    'gpt-4.1-mini': 0.4,
    'gpt-4.1-nano': 0.1,
    'deepseek-chat': 0.28,
    'deepseek-reasoner': 0.28,
    'Llama3-8B': 0,
    'Qwen3-14B': 0,
    'Qwen3-14B-Thinking': 0,
    'Gemma3-4B': 0,
    'Apertus-8b': 0,
    'Apertus-70b': 0,
}

OUTPUT_COST_DICT = {
    'GPT-OSS-20B': 0,
    'GPT-OSS-120B': 0,
    'GPT-OSS-120B-medium': 0,
    'GPT-OSS-20B-medium': 0,
    'llama3-8b': 0,
    'llama3-70b': 0,
    'gemma3-27b': 0,
    'Qwen3-30B': 0,
    'Qwen3-8B': 0,
    'Qwen3-8B-Thinking': 0,
    'Qwen3-Next': 0,
    'Qwen3-Next-Thinking': 0,
    'Qwen3-32B': 0,
    'Qwen3-32B-Thinking': 0,
    'gpt-5': 10,
    'gpt-5-mini': 2,
    'gpt-5-nano': 0.4,
    'gpt-4.1': 8,
    'gpt-4.1-mini': 1.6,
    'gpt-4.1-nano': 0.4,
    'deepseek-chat': 0.42,
    'deepseek-reasoner': 0.42,
    'Llama3-8B': 0,
    'Qwen3-14B': 0,
    'Qwen3-14B-Thinking': 0,    
    'Gemma3-4B': 0,
    'Apertus-8b': 0,
    'Apertus-70b': 0,
}

GENE_ARGS_DICT = {
    'GPT-OSS-20B': {
        'extra_body': {"reasoning_effort": 'high'},
    },
    'GPT-OSS-120B': {
        'extra_body': {"reasoning_effort": 'high'},
    },
    'GPT-OSS-120B-medium': {
        'extra_body': {"reasoning_effort": 'medium'},
    },
    'GPT-OSS-20B-medium': {
        'extra_body': {"reasoning_effort": 'medium'},
    },
    'Qwen3-30B': {
        'temperature': 0.6,
        'top_p': 0.95,
    },
    'Qwen3-Next-Thinking': {
        'temperature': 0.7,
        'top_p': 0.8,
        'extra_body': {"top_k": 20},
    },
    'Qwen3-Next': {
        'temperature': 0.7,
        'top_p': 0.8,
        'extra_body': {"top_k": 20},
    },
    'Qwen3-32B': {
        'max_tokens': 10240,
        'temperature': 0.6,
        'top_p': 0.95,
        'extra_body': {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 20},
    },
    'Qwen3-32B-Thinking': {
        'max_tokens': 10240,
        'temperature': 0.6,
        'top_p': 0.95,
        'extra_body': {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20},
    },
    'Qwen3-8B-Thinking': {
        'temperature': 0.6,
        'top_p': 0.95,
        'extra_body': {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20},
    },
    'llama3-8b': {
        'temperature': 0,
    },
    'Qwen3-8B': {
        'temperature': 0,
        'extra_body': {"chat_template_kwargs": {"enable_thinking": False}},
    },
    'llama3-70b': {
        'temperature': 0,
    },
    'gemma3-27b': {
        'temperature': 0,
    },
    'Qwen3-14B': {
        'temperature': 0,
        'extra_body': {"chat_template_kwargs": {"enable_thinking": False}},
    },
    'Qwen3-14B-Thinking': {
        'temperature': 0.6,
        'top_p': 0.95,
        'extra_body': {"chat_template_kwargs": {"enable_thinking": True}, "top_k": 20},
    },
    'gpt-5': {'reasoning_effort': 'medium'},
    'gpt-5-mini': {'reasoning_effort': 'medium'},
    'gpt-5-nano': {'reasoning_effort': 'medium'},
    'gpt-4.1': {'temperature': 0},
    'gpt-4.1-mini': {'temperature': 0},
    'gpt-4.1-nano': {'temperature': 0},
    'deepseek-reasoner': {'temperature': 0.6},
    'deepseek-chat': {'temperature': 0},
    'Llama3-8B': {'temperature': 0},
    'Gemma3-4B': {'temperature': 0},
    'Apertus-8b': {'temperature': 0},
    'Apertus-70b': {'temperature': 0},
}


def get_input_price(model, input_len=None):
    input_cost = input_len / 1000000 * INPUT_COST_DICT[model]
    return input_cost


def get_output_price(model, output_len=None):
    output_cost = output_len / 1000000 * OUTPUT_COST_DICT[model]
    return output_cost



async def achat(model, messages, generation_args, client, get_logprobs=False, top_logprobs=30):
    api_kwargs = generation_args.copy()
    if get_logprobs:
        # Use logprobs as a number (e.g., 30) to request top-30 token probabilities
        api_kwargs['logprobs'] = get_logprobs
        api_kwargs['top_logprobs'] = top_logprobs
    
    output = await client.chat.completions.create(
        model=MODEL_DICT[model],
        messages=messages,
        **api_kwargs
    )
    input_token_num = output.usage.prompt_tokens
    output_token_num = output.usage.completion_tokens
    try:
        reasoning_content = output.choices[0].message.reasoning_content
    except Exception:
        reasoning_content = None
    
    # Extract logprobs if available
    if get_logprobs:
        logprobs_data = None
        choice = output.choices[0]
        # For chat completions, logprobs are available on choice.logprobs
        if hasattr(choice, 'logprobs') and choice.logprobs and hasattr(choice.logprobs, 'content'):
            tokens = []
            token_logprobs = []
            top_logprobs_list = []
            text_offsets = []
            current_offset = 0

            for token_info in choice.logprobs.content:
                token_str = getattr(token_info, 'token', None)
                token_lp = getattr(token_info, 'logprob', None)
                top_entries = getattr(token_info, 'top_logprobs', None)

                tokens.append(token_str)
                token_logprobs.append(float(token_lp) if token_lp is not None else None)

                if top_entries:
                    top_dict = {}
                    for top_entry in top_entries:
                        top_token = getattr(top_entry, 'token', None)
                        top_lp = getattr(top_entry, 'logprob', None)
                        if top_token is not None and top_lp is not None:
                            top_dict[top_token] = float(top_lp)
                    top_logprobs_list.append(top_dict)
                else:
                    top_logprobs_list.append(None)

                text_offsets.append(current_offset)
                if token_str is not None:
                    current_offset += len(token_str)

            logprobs_data = {
                'tokens': tokens,
                'token_logprobs': token_logprobs,
                'top_logprobs': top_logprobs_list,
                'text_offset': text_offsets,
            }
        return output.choices[0].message.content, reasoning_content, input_token_num, output_token_num, logprobs_data
    else:
        return output.choices[0].message.content, reasoning_content, input_token_num, output_token_num


def batchify(lst, batch_size):
    """Split the list `lst` into sublists of size `batch_size`."""
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


async def create_answers_async(model, messages, cache_path, generation_args, client, batch_size=5, get_logprobs=False, top_logprobs=30):
    # async answering
    batched_msgs = batchify(messages, batch_size)
    total_input_tok_num = 0
    total_output_tok_num = 0
    print("{} batches to run.".format(len(batched_msgs)))
    all_answers = []
    all_reasoning_contents = []
    all_logprobs = []
    cache_settings = dc.DEFAULT_SETTINGS.copy()
    cache_settings["eviction_policy"] = "none"
    cache_settings["size_limit"] = int(1e12)
    cache_settings["cull_limit"] = 0
    error_batches = []
    with dc.Cache(cache_path, **cache_settings) as litellm_responses:
        for i, batch in tqdm(enumerate(batched_msgs), total=len(batched_msgs)):
            mapping_list = []
            cache_miss_msgs = []
            cache_hit_responses = []
            cache_hit_cots = []
            cache_hit_logprobs = []
            for msg_in_batch in batch:
                cache_key = (model, msg_in_batch, get_logprobs, top_logprobs) if get_logprobs else (model, msg_in_batch)
                if cache_key in litellm_responses:
                    mapping_list.append(len(cache_hit_responses) + 1)
                    cache_hit_responses.append(litellm_responses[cache_key]['response'])
                    cache_hit_cots.append(litellm_responses[cache_key]['response_reasoning'])
                    if get_logprobs:
                        cache_hit_logprobs.append(litellm_responses[cache_key].get('logprobs', None))
                else:
                    mapping_list.append(- len(cache_miss_msgs) - 1)
                    cache_miss_msgs.append(msg_in_batch)

            if len(cache_miss_msgs) == 0:
                all_answers.extend(cache_hit_responses)
                all_reasoning_contents.extend(cache_hit_cots)
                if get_logprobs:
                    all_logprobs.extend(cache_hit_logprobs)
                print(f"Batch {i} entirely Loaded")
            else:
                try:
                    api_responses = await asyncio.gather(*[achat(model, m, generation_args, client, get_logprobs, top_logprobs) for m in cache_miss_msgs])
                    if get_logprobs:
                        answers, reasoning_contents, input_tok_nums, output_tok_nums, logprobs_list = zip(*api_responses)
                    else:
                        answers, reasoning_contents, input_tok_nums, output_tok_nums = zip(*api_responses)
                        logprobs_list = [None] * len(answers)
                    total_input_tok_num += sum(input_tok_nums)
                    total_output_tok_num += sum(output_tok_nums)
                    for msg, res, reasoning, logprobs in zip(cache_miss_msgs, answers, reasoning_contents, logprobs_list):
                        cache_key = (model, msg, get_logprobs, top_logprobs) if get_logprobs else (model, msg)
                        litellm_responses[cache_key] = {
                            'response': res, 
                            'response_reasoning': reasoning,
                            'logprobs': logprobs if get_logprobs else None
                        }
                    merged_responses = []
                    merged_reasoning_contents = []
                    merged_logprobs = []
                    for idx in mapping_list:
                        if idx > 0:
                            merged_responses.append(cache_hit_responses[idx - 1])
                            merged_reasoning_contents.append(cache_hit_cots[idx - 1])
                            if get_logprobs:
                                merged_logprobs.append(cache_hit_logprobs[idx - 1])
                        else:
                            merged_responses.append(answers[- idx - 1])
                            merged_reasoning_contents.append(reasoning_contents[- idx - 1])
                            if get_logprobs:
                                merged_logprobs.append(logprobs_list[- idx - 1])
                    all_answers.extend(merged_responses)
                    all_reasoning_contents.extend(merged_reasoning_contents)
                    if get_logprobs:
                        all_logprobs.extend(merged_logprobs)
                    print(f"Batch {i} Done")
                except Exception as e:
                    print(f"Batch {i} Error while gathering answers: {e}")
                    error_batches.append(i)

    input_price = get_input_price(model, total_input_tok_num)
    output_price = get_output_price(model, total_output_tok_num)
    if get_logprobs:
        return all_answers, all_reasoning_contents, all_logprobs, error_batches, input_price + output_price
    else:
        return all_answers, all_reasoning_contents, error_batches, input_price + output_price
