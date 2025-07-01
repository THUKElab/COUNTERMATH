import os
import json
from typing import Optional, List
import time
import jsonlines
from tqdm import tqdm
from openai import OpenAI
import fire
import requests
from datetime import datetime
import random

# Default settings for Evaluator
MAX_TOKENS = 4096
OPENAI_MODEL = 'gpt-4o'
TEMPERATURE = 0.7
TOP_P: float = 1.0
NUM_RETURN_SEQUENCES: int = 1
RATE_LIMIT_PER_MIN: Optional[int] = None
STOP: Optional[str] = None
LOGPROBS: Optional[int] = 0

client = OpenAI(
    api_key = json.load(open('config/api_config.json'))['openai']['api_key'],
    base_url= 'URL'
)

# 添加余额查询函数
def check_balance(api_key: str) -> dict:
    '''查询API余额'''
    # try:
    #     url = 'https://billing.openkey.cloud/api/token'
    #     response = requests.post(url, json={'api_key': api_key})
    #     if response.status_code == 200:
    #         return response.json()
    #     return {"Status": 0, "Error": f"请求失败: {response.status_code}"}
    # except Exception as e:
    #     return {"Status": 0, "Error": f"发生异常: {str(e)}"}
    return {"Status": 0, "Error": "发生异常: "}

def generate(prompt):
    ''' generation using OpenAI API '''
    retry_count = 0
    while retry_count < 5:
        try:
            if RATE_LIMIT_PER_MIN is not None:
                time.sleep(60 / RATE_LIMIT_PER_MIN)
            if ('gpt-3.5-turbo' in OPENAI_MODEL) or ('gpt-4' in OPENAI_MODEL):
                messages = [{'role': 'user', 'content': prompt}]
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    n=NUM_RETURN_SEQUENCES,
                    stop=STOP,
                )
                text=[choice.message.content for choice in response.choices]
                return text
            else:
                raise ValueError(f'Unknown OPENAI MODEL {OPENAI_MODEL}')
        except Exception as e:
            print(f'An Error Occured: {e}, sleeping for 5 seconds')
            time.sleep(5)
            retry_count += 1
    return "Request Fail"

def autorace_score(output_log_path:str):
    '''report autorace score'''
    #load the autorace evaluation
    with jsonlines.open(output_log_path, mode='r') as reader:
        autorace = list(reader)

    #calculate the score
    total = len(autorace)
    incorrect = 0
    for i in range(total):
        if 'INCORRECT' in autorace[i]['evaluation_result'][0]:
            incorrect += 1
    print(f'output_log_path: {output_log_path}')
    print(f'total: {total}, incorrect: {incorrect}')
    print(f'autorace score: {(total - incorrect) / total:.4f}')
    print(f'==============================================\n\n')

def split_steps(model_output: str) -> List[str]:
    """将模型输出按步骤分割"""
    if not model_output:
        return []
    
    # 处理不同的分隔符格式
    steps = []
    if "### Step" in model_output:
        steps = model_output.split("### Step")
    elif "###Step" in model_output:
        steps = model_output.split("###Step")
    elif "### step" in model_output:
        steps = model_output.split("### step") 
    elif "###step" in model_output:
        steps = model_output.split("###step")
    else:
        steps = [model_output]  # 如果没有找到分隔符,将整个输出作为一个步骤
    
    # 清理步骤内容
    cleaned_steps = []
    for step in steps:
        # 去除空白步骤
        if not step.strip():
            continue
        
        # 清理步骤内容
        step_content = step.strip()
        # 如果步骤以数字和点开头，移除它们
        if step_content[0].isdigit():
            step_content = step_content.split(":", 1)[-1].strip()
        
        cleaned_steps.append(step_content)
    
    return cleaned_steps

def evaluate_steps(
    steps: List[str],
    prompt_template: str,
    statement: str,
    prediction: str,
    answer: str,
    rationale: str,
    chunk_size: int = 3
) -> List[dict]:
    """评估每个步骤组合，按chunk_size进行分组评估"""
    evaluations = []
    current_steps = []
    total_steps = len(steps)
    print(f"Number of steps: {total_steps}")
    
    # 计算需要评估的次数
    num_chunks = (total_steps + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        # 计算当前chunk的结束位置
        end_idx = min((chunk_idx + 1) * chunk_size, total_steps)
        
        # 添加这个chunk中的所有步骤
        while len(current_steps) < end_idx:
            step_idx = len(current_steps)
            current_steps.append(steps[step_idx])
        
        # 组合所有当前步骤
        combined_steps = "Step 1: " + current_steps[0]  # 确保第一步有正确的格式
        if len(current_steps) > 1:
            # 从第二步开始添加其余步骤
            combined_steps += "\n" + "\n".join(
                f"Step {i+2}: {step}" 
                for i, step in enumerate(current_steps[1:])
            )
            
        print(f"\n正在评估Chunk {chunk_idx + 1} (Steps 1-{end_idx}):\n{combined_steps}")
        
        # 构造评估prompt
        prompt = prompt_template.format(
            statement=statement,
            reasoning_steps=combined_steps,
            rationale=rationale,
        )
        print(prompt)
        
        # 获取评估结果
        evaluation = generate(prompt)[0]
        
        evaluations.append({
            'steps_included': end_idx,
            'evaluation': evaluation,
            'prompt': prompt,
            'steps_evaluated': combined_steps
        })
        
        # 添加随机延迟以避免速率限制
        time.sleep(random.uniform(1, 3))
    
    return evaluations

def evaluate_single_item(item: dict, prompt_type: str, prompt_template: str, only_score: bool = False) -> dict:
    """评估单个样本"""
    # 构造评估prompt
    try:
        statement = item['statement']
        rationale = item['rationale'] if 'rationale' in item else item['chinese_rationale']
        judgement = item['answer']
        model_output = item['model_output']
        prediction = item['prediction']
        # proof_output = data[index]['proof_output']
    except Exception as e:
        print(f'An Error Occured: {e}')
        print("==================")
        print(item)
        return
    try:
        if prompt_type == 'justify_with_ref' or prompt_type == 'justify_with_ref_zh':
            formatted_prompt = prompt_template.format(statement, judgement, rationale, prediction, model_output)
        elif prompt_type == 'justify_auto':
            formatted_prompt = prompt_template.format(statement, judgement, model_output)
        else:
            raise ValueError(f'Unknown prompt type {prompt_type}')
    except Exception as e:
        print(f'An Error Occured: {e}')
        print("==================")
        print(formatted_prompt)
        print("==================")
        print("statement: ", statement)
        print("rationale: ", rationale)
        print("judgement: ", judgement)
        print("verify_output: ", model_output)
        return {
            'id': item.get('id', None),
            'statement': item['statement'],
            'answer': item['answer'],
            'model_output': item.get('model_output', ''),
            'evaluation_result': "ERROR"
        }
    
    # 获取评估结果
    evaluation = generate(formatted_prompt)[0]
    
    return {
        'id': item.get('id', None),
        'statement': item['statement'],
        'answer': item['answer'],
        'rationale': rationale,
        'prediction': prediction,
        'model_output': item.get('model_output', ''),
        'evaluation_result': [evaluation],
        'prompt': formatted_prompt

    }

def autorace_evaluation(
    file_path: str,
    log_path: str,
    prompt_type: str = 'justify_with_ref',
    only_score: bool = False,
    output_log_path: str = None
):
    if only_score and output_log_path is not None:
        autorace_score(output_log_path)
        return
    elif only_score and output_log_path is None:
        raise ValueError('only_score and output_log_path must be provided together')
    
    # 在开始时检查余额
    api_key = json.load(open('config/api_config.json'))['openai']['api_key']
    initial_balance = check_balance(api_key)
    initial_amount = initial_balance.get('Remaining', 0)
    
    os.makedirs(log_path, exist_ok=True)
    input_filename = os.path.splitext(os.path.basename(file_path))[0]
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    
    if output_log_path is None:
        output_log_path = f'{log_path}/autorace_eval_{input_filename}_{prompt_type}_{current_time}.jsonl'
    
    prompt = json.load(open('evaluation_prompt.json'))[prompt_type]
    results = []
    total_lines = sum(1 for _ in jsonlines.open(file_path))
    
    Incorrect_num = 0
    correct_num = 0

    with tqdm(total=total_lines) as pbar:
        for idx, item in enumerate(jsonlines.open(file_path)):
            # 检查是否需要进行步骤评估
            if prompt_type.startswith('step_by_step') and 'model_output_disassemble' in item:
                steps = split_steps(item['model_output_disassemble'])
                if steps:
                    step_evaluations = evaluate_steps(
                        steps,
                        prompt,
                        item['statement'],
                        item['prediction'],
                        item['answer'],
                        item['rationale']
                    )
                    result = {
                        'id': item.get('id', None),
                        'statement': item['statement'],
                        'answer': item['answer'],
                        'prediction': item['prediction'],
                        'rationale': item['rationale'],
                        'model_output': item['model_output'],
                        'model_output_disassemble': item['model_output_disassemble'],
                        'step_evaluations': step_evaluations
                    }
                else:
                    # 如果没有步骤，使用原始评估
                    result = evaluate_single_item(item, 'justify_with_ref', prompt, only_score)
            else:
                # 使用原始评估
                result = evaluate_single_item(item, prompt_type, prompt, only_score)
            
            results.append(result)
            if not prompt_type.startswith("step_by_step"):
                evaluation_result = result['evaluation_result'][0]
                if 'INCORRECT' in evaluation_result:
                    Incorrect_num += 1
                else:
                    correct_num += 1
                tqdm.write(f"INCORRECT: {Incorrect_num}, CORRECT: {correct_num}, TOTAL: {idx + 1}, "
                  f"ACCURACY: {correct_num / (idx + 1):.4f}")
            
            # 每10个样本检查一次余额并更新文件名
            if idx % 10 == 0:
                current_balance = check_balance(api_key)
                current_amount = current_balance.get('Remaining', 0)
                cost_so_far = initial_amount - current_amount
                
                # 更新文件名
                new_output_log_path = f'{log_path}/autorace_eval_{input_filename}_{prompt_type}_cost{cost_so_far:.2f}_{current_time}.jsonl'
                if os.path.exists(output_log_path):
                    os.rename(output_log_path, new_output_log_path)
                output_log_path = new_output_log_path
                
                tqdm.write(f"当前花费: ${cost_so_far:.5f}")
            
            # 保存当前结果
            with jsonlines.open(output_log_path, mode='w') as writer:
                writer.write_all(results)
            
            pbar.update(1)

    # 计算总费用
    final_balance = check_balance(api_key)
    final_amount = final_balance.get('Remaining', 0)
    total_cost = initial_amount - final_amount
    
    print(f"\n总费用: ${total_cost:.4f}")
    print(f"初始余额: ${initial_amount:.4f}")
    print(f"最终余额: ${final_amount:.4f}")

def main(
    file_path: str='outputPATH',
    log_path: str='output/log',
    prompt_type: str='justify_with_ref',
    only_score: bool=False, 
    output_log_path: str=None
):
    autorace_evaluation(file_path, log_path, prompt_type, only_score, output_log_path)

if __name__ == '__main__':
    fire.Fire(main)
