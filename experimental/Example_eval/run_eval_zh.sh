#!/bin/bash

# 定义需要评估的文件列表
FILES=(
    "FILEPATH"
)

# 遍历文件列表
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Evaluating: $file"
        python src/eval_rationale.py -f "$file" --prompt_type justify_with_ref_zh --log_path output/log_zh
        echo "----------------------------------------"
    else
        echo "No file: $file"
    fi
done

echo "Complete!"