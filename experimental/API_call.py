import requests
import json
import re
import os

API_URL = "YOUR_URL"
API_KEY = "YOUR_API_KEY"

headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {API_KEY}"
    }

def extract_answer(text):
    pattern = r'\\boxed\{(.*?)\}'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return ''

### english prompt
instr_en = "Please reason step by step about whether the statement is True or False, and put your final answer within \boxed{}."

### chinese prompts
instr = "请通过逐步推理来判断命题为“真”或“假”，并把最终答案放置于\boxed{}中。"

def response_countermath(input_data, output_file='output_gpt_chinese_again_1.json'):
    for item in input_data:
        prompt = (
            f"{item['chinese_statement']}"
        )
        # call API
        response = requests.post(API_URL, headers=headers, json={
            "model": "Model",
            "messages": [{"role": "system", "content": instr},{"role": "user", "content": prompt}],
            "max_tokens": 2048,  
            "temperature": 0.7,
        })

        # check
        if response.status_code == 200:
            result = response.json()
            response_content = result['choices'][0]['message']['content'].strip()  
            print(response_content)
            
            usage_info = result.get('usage', {})
            used_tokens = usage_info.get('total_tokens', 'unknown')  
            extracted_answer = extract_answer(response_content)

            # JSON
            output_entry = {
                "model_output": response_content,
                "prediction": extracted_answer,
                "used_tokens": used_tokens
            }
            print(output_entry)
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(output_entry, ensure_ascii=False))
                f.write('\n')  
        else:
            print(f"Error: {response.status_code}, {response.text}")

with open('./countermath_ver1.1.jsonl', 'r') as f:
    lines = f.readlines()
    input_data = [json.loads(line) for line in lines]

response_countermath(input_data)
