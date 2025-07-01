MODEL_NAME=Model
# defaul_max_tokens = 512
CUDA_VISIBLE_DEVICES=0 python baseline-eval.py \
    --save_dir results-Countermath/$MODEL_NAME-2048 \
    --model_name Model \
    --clm_max_length 2048 \
    --additional_stop_sequence "<|im_end|>"