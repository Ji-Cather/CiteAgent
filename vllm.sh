# python -m vllm.entrypoints.openai.api_server \
#     --model /data/jiarui_ji/Meta-Llama-3-8B-Instruct \
#     --dtype half \
#     --port 8000 \
#     --host 127.0.0.1 \
#     --num-gpus 1 \
#     --max-batch-size 8 \
#     --max-seq-len 8192 \
#     --log-requests \

export OPENAI_API_KEY="sk-api"

python -m vllm.entrypoints.openai.api_server \
  --model /data/jiarui_ji/Meta-Llama-3-8B-Instruct \
  --served-model-name llama3_8b \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --port 8001 \
  --host 127.0.0.1 \
  --api-key $OPENAI_API_KEY
