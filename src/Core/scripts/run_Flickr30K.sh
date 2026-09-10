# ⚠️ 暂不可运行：prompts/Flickr30K/ 模板缺失，main.py 启动时会报错退出。
#    补齐 prompts/Flickr30K/ 下的 4 个必需模板后方可使用。
python main.py --split test --dataset Flickr30K --run_name flickr30k\
 --llm_path "Reasoner LLM model path"\
 --mllm_path "Verifier MLLM model path"\
 --img_cap_model_path "Captioner MLLM model path"\
 --eval_mllm_path "Evalutor MLLM model path"\
 --clip_path "Your VLM model path"\
 --dataset_path "Dataset folder path"