# serve/serve.py

import subprocess
import sys

def start_server():
    cmd = [
        sys.executable,
        "-m", "vllm.entrypoints.openai.api_server",
        "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "--port", "8000",
        "--host", "0.0.0.0",
        "--max-model-len", "1024",
        "--dtype", "float32",  # better CPU compatibility
    ]

    subprocess.run(cmd)

if __name__ == "__main__":
    start_server()