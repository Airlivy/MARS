# ⚠️ 暂不可运行：prompts/VisDial/ 模板缺失，main.py 启动时会报错退出。
#    补齐 prompts/VisDial/ 下的 4 个必需模板后方可使用。
python main.py --split valid --dataset VisDial --run_name visdial\
 --llm_path "Reasoner LLM model path"\
 --mllm_path "Verifier MLLM model path"\
 --img_cap_model_path "Captioner MLLM model path"\
 --eval_mllm_path "Evalutor MLLM model path"\
 --clip_path "Your VLM model path"\
 --dataset_path "Dataset folder path"