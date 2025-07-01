# 遍历output/log目录下的所有文件，并展示结果
for file in output/log/*; do
    python eval_rationale.py --only_score --output_log_path "$file"
done
