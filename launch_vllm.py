import subprocess
import time

# Launch vLLM as a completely detached background process
process = subprocess.Popen(
    "VLLM_USE_V2_MODEL_RUNNER=0 python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B  --port 8000 --gpu_memory_utilization 0.8 --max-model-len 4096 > vllm.log 2>&1 &",
    shell=True
)

print("Starting vLLM server in background...")
time.sleep(30) # Wait for weights to load
print("vLLM should be ready now! you can check if vllm running via curl http://localhost:8000/v1/models")
