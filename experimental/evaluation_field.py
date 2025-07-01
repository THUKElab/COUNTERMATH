import json
from sklearn.metrics import accuracy_score, f1_score

keys_to_keep = ["statement", "answer", "prediction", "model_output", "rationale", "field"]

a_file_path = 'predictions.jsonl'
with open(a_file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    data_a = [json.loads(line) for line in lines]

b_file_path = 'countermath_ver1.1.jsonl'
with open(b_file_path, 'r', encoding='utf-8') as f:
    data_b = [json.loads(line) for line in f]


if len(data_a) != len(data_b):
    print("A:", len(data_a))
    print("B:", len(data_b))
    raise ValueError("Wrong merge!")

for i in range(len(data_a)):
    statement = data_b[i].get('statement')
    field = data_b[i].get('field')
    rationale = data_b[i].get('rationale')
    answer = data_b[i].get('judgement')

    if statement is not None:
        data_a[i]['statement'] = statement
    if field is not None:
        data_a[i]['field'] = field
    if rationale is not None:
        data_a[i]['rationale'] = rationale
    if answer is not None:
        data_a[i]['answer'] = answer

filtered_data = [{k: item[k] for k in keys_to_keep if k in item} for item in data_a]

output_file = 'merged.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)

def f1_answer_mapping(_list):
    mappings = {'False': 0, 'True': 1, '': 2}
    return [mappings.get(x, 2) for x in _list]  

input_file_path = output_file 
evaluation_result_path = 'evaluation_results.json'

with open(input_file_path, 'r', encoding='utf-8') as f:
    filtered_data = json.load(f)


field_metrics = {}


for entry in filtered_data:
    field = entry.get('field', 'default')  
    
    if field not in field_metrics:
        field_metrics[field] = {'answers': [], 'predictions': []}
    
    answer = str(entry['answer']).capitalize() 
    prediction = str(entry['prediction']).capitalize()  
    
    field_metrics[field]['answers'].append(answer)
    field_metrics[field]['predictions'].append(prediction)

for field in field_metrics.keys():
    field_metrics[field]['answers'] = f1_answer_mapping(field_metrics[field]['answers'])
    field_metrics[field]['predictions'] = f1_answer_mapping(field_metrics[field]['predictions'])

results = {}
for field, metrics in field_metrics.items():
    accuracy = accuracy_score(metrics['answers'], metrics['predictions'])
    macro_f1 = f1_score(metrics['answers'], metrics['predictions'], average='macro')
    
    results[field] = {
        "length_of_data": len(metrics['answers']),
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }

    print(f"Field: {field}")
    print(f"Length of data: {len(metrics['answers'])}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1 Score: {macro_f1:.4f}")
    print()

all_answers = [item for sublist in [metrics['answers'] for metrics in field_metrics.values()] for item in sublist]
all_predictions = [item for sublist in [metrics['predictions'] for metrics in field_metrics.values()] for item in sublist]

overall_accuracy = accuracy_score(all_answers, all_predictions)
overall_macro_f1 = f1_score(all_answers, all_predictions, average='macro')

results['overall'] = {
    "length_of_data": len(all_answers),
    "accuracy": overall_accuracy,
    "macro_f1": overall_macro_f1
}

print("Overall Metrics")
print(f"Length of data: {len(all_answers)}")
print(f"Accuracy: {overall_accuracy:.4f}")
print(f"Macro-F1 Score: {overall_macro_f1:.4f}")

with open(evaluation_result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results have been saved to {evaluation_result_path}")