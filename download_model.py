from huggingface_hub import snapshot_download
import os

print("Downloading Mistral-7B...")

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    local_dir="/model-cache",
    ignore_patterns=["*.pt"],  # skips old format, saves ~2GB
)

print("✅ Model downloaded to /model-cache")