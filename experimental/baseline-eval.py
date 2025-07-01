import random
import json
import vllm
import evaluate
import argparse
from tqdm import tqdm
import torch
import os
import re
from transformers import AutoTokenizer

exact_match = evaluate.load("./evaluation/metrics/exact_match")
f1 = evaluate.load('./evaluation/metrics/f1')

origin_prompt = lambda x: f"""Please judge whether the following statement is True or False: 

{x}

You should think step by step in the Thought and give your judgement, i.e., True or False, after Judgement.
Thought: Let's think step by step.
Judgement: """

# Vanilla prompt
### completion prompts
completion_prompt = lambda x: f"{x}\n" + "Please reason step by step about whether the above statement is True or False, and put your final answer within \\boxed{}."
abel_prompt = lambda x: f"Question: \n{x}"+"\nAnswer: \nLet's think step by step about whether the answer is True or False, and put your final answer within \\boxed{}.\n"
hint_completion_prompt = lambda x: f"{x}\n" + "Please reason by giving examples about whether the above statement is True or False, and put your final answer within \\boxed{}."
### instruct prompts
instruct_prompt = lambda x: f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{completion_prompt(x)}\n\n### Response:"""

### chat prompts
instr = "Please reason step by step about whether the statement is True or False, and put your final answer within \\boxed{}."
chat_prompt = lambda x: [{"role": "system", "content": instr}, {"role": "user", "content": x}]
numina_prompt = lambda x: [{'role': 'user', 'content': completion_prompt(x)}]

# Hint prompt
### completion prompts
completion_prompt = lambda x: f"{x}\n" + "Please reason by giving examples about whether the above statement is True or False, and put your final answer within \\boxed{}."
abel_prompt = lambda x: f"Question: \n{x}"+"\nAnswer: \nPlease reason by giving examples about whether the above statement is True or False, and put your final answer within \\boxed{}.\n"
hint_completion_prompt = lambda x: f"{x}\n" + "Please reason by giving examples about whether the above statement is True or False, and put your final answer within \\boxed{}."
### instruct prompts
instruct_prompt = lambda x: f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{completion_prompt(x)}\n\n### Response:"""

### chat prompts
instr = "Please reason by giving examples about whether the above statement is True or False, and put your final answer within \\boxed{}."
chat_prompt = lambda x: [{"role": "system", "content": instr}, {"role": "user", "content": x}]
numina_prompt = lambda x: [{'role': 'user', 'content': completion_prompt(x)}]


def extract_answer(text):
    splits = text.strip().split('.')
    first_sent = splits[0]
    pattern = r"the statement is (true|false)"
    last_sent = ''
    for i in range(len(splits)-1, -1, -1):
        if splits[i]:
            last_sent = splits[i]
            break
    if first_sent:
        # match = re.search(pattern, first_sent.lower(), re.IGNORECASE)
        match = bool(re.fullmatch(pattern, first_sent.lower()))
        if match:
            if 'true' in first_sent.lower():
                return 'True'
            elif 'false' in first_sent.lower():
                return 'False'
    if last_sent:
        if 'true' in last_sent.lower():
            return 'True'
        elif 'false' in last_sent.lower():
            return 'False'
    return ''


def f1_answer_mapping(_list):
    mappings = {'False': 0, 'True': 1, '': 2}
    return list(map(lambda x: mappings[x], _list))


def main(args):
    random.seed(42)
    
    is_chat = False
    lower_model_name = args.model_name.lower()
    # use the proper prompt!
    if any([i in lower_model_name for i in ['wizard']]):
        prompt = instruct_prompt
    elif any([i in lower_model_name for i in ['qwen', 'numina', 'internlm', 'mathstral']]):
        if 'numina' in lower_model_name:
            prompt = numina_prompt
        else:
            prompt = chat_prompt
        is_chat = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        add_generation_prompt = True
    else:
        if 'abel' in lower_model_name:
            prompt = abel_prompt
        else:
            prompt = completion_prompt

    print("Loading data...")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with open('countermath_ver1.1.jsonl', 'r') as f:
        lines = f.readlines()
        full_data = [json.loads(line) for line in lines]
    
    if args.num_instances is not None and len(full_data) >= args.num_instances:
        full_data = random.sample(full_data, args.num_instances)

    prompts = []
    targets = []
    rationales = []
    if args.eval_lang == 'chn':
        for d in tqdm(full_data):
            prp = prompt(d['chinese_statement'])
            if is_chat:
                prp = tokenizer.apply_chat_template(
                    prp, 
                    tokenize=False, 
                    add_generation_prompt=add_generation_prompt
                )
            prompts.append(prp)
            targets.append(str(d['judgement']))
            rationales.append(d['chinese_rationale'])
    else:
        for d in tqdm(full_data):
            prp = prompt(d['statement'])
            if is_chat:
                prp = tokenizer.apply_chat_template(
                    prp, 
                    tokenize=False, 
                    add_generation_prompt=add_generation_prompt
                )
            prompts.append(prp)
            targets.append(str(d['judgement']))
            rationales.append(d['rationale'])

    # load model and tokenizer
    model = vllm.LLM(
        model=args.model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        # max_model_len=32960, # for internlm2-math-plus-mixtral8x22b
        # gpu_memory_utilization=.75
    )

    stop_strings = args.additional_stop_sequence
    if args.newline_stop:
        if args.stop_at_double_newline:
            stop_strings += ["\n\n"] 
        elif args.stop_at_triple_newline:
            stop_strings += ["\n\n\n"]
        else:
            stop_strings += ["\n"]
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=args.clm_max_length,
        stop=stop_strings,
        skip_special_tokens=True,
    )

    # We need to remap the outputs to the prompts because vllm might not return outputs for some prompts (e.g., if the prompt is too long)
    generations = model.generate(prompts, sampling_params)
    prompt_to_output = {
        g.prompt: g.outputs[0].text for g in generations
    }
    outputs = [prompt_to_output[prompt] if prompt in prompt_to_output else "" for prompt in prompts]
    # print(prompt_to_output)
    # print("Gold results== ", targets[0])
    # print(rationales[0])
    # print('extracted answer== ', extract_answer(outputs[0]))
    # exit(0)

    print("Calculating accuracy...")
    predictions = []
    for output in outputs:
        answer = extract_answer(output)
        if answer:
            predictions.append(answer)
        else:
            predictions.append("")
        
    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    # f1_score = f1.compute(predictions=f1_answer_mapping(predictions), references=f1_answer_mapping(targets), average='macro')['f1']
    print(f"Exact match : {em_score}")
    # print(f"F1 Score : {f1_score}")

    predictions = [{
        "prompt": prompt,
        "answer": tgt,
        "prediction": pred,
        "model_output": output,
        "rationale": d['rationale'],
        "field": d['field']
    } for prompt, tgt, output, pred, d in zip(prompts, targets, outputs, predictions, full_data)]

    with open(os.path.join(args.save_dir, f"predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n") 
    
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score,
            # "f1_score": f1_score
        }, fout, indent=4)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name', 
    type=str, 
    help="The HuggingFace model to be evaluated."
    )
parser.add_argument(
    '--num_instances', 
    type=int, 
    default=None,
    help="Num of sampled instances for evaluation"
    )
parser.add_argument(
    "--newline_stop",
    action="store_true",
    help="If given, we will use stop token (usually newline or double newline) to stop generation."
    )
parser.add_argument(
    "--stop_at_double_newline",
    action="store_true",
    help="If given, will stop generation at double newline instead of single."
    )
parser.add_argument(
    "--stop_at_triple_newline",
    action="store_true",
    help="If given, will stop generation at triple newline instead of single."
    )
parser.add_argument(
    '--additional_stop_sequence',
    type=str,
    nargs="+",
    default=[],
    help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
parser.add_argument(
    "--clm_max_length",
    type=int,
    default=256
    )
parser.add_argument(
    "--eval_lang",
    type=str,
    choices=['chn', 'eng'],
    default='eng'
    )
parser.add_argument(
        "--save_dir", 
        type=str
    )


args = parser.parse_args()
main(args)