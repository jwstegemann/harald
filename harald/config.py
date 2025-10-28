import modal
import os
from huggingface_hub import hf_hub_download
from pathlib import Path

# ----------------------------------------------------------------------
#  Parameter
# ----------------------------------------------------------------------
MAX_PER_ID = 5
IMAGE_SIZE = 448


# ----------------------------------------------------------------------
#  modal - Infra
# ----------------------------------------------------------------------

bucket_creds = modal.Secret.from_name("aws-secret", environment_name="main")

raw_bucket_name = "cida-datasets-raw"
target_bucket_name = "cida-datasets-target"

raw_volume = modal.CloudBucketMount(
    raw_bucket_name,
    secret=bucket_creds,
)

target_volume = modal.CloudBucketMount(
    target_bucket_name,
    secret=bucket_creds,
)

RAW_DIR = "/mnt/raw"
TARGET_DIR = "/mnt/target"

HF_CACHE = Path("/models")  # shared HF cache
HF_SECRET = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim()
    .apt_install(
        "wget",
        "git",
        "git-lfs",
        "ca-certificates",
        "build-essential",
        "libxrender-dev",
        "curl",
        "file",
        "ffmpeg",
        "libsm6",
        "libxext6",
        "libgl1-mesa-glx",
        "libglib2.0-0",
    )
    .run_commands(
        "wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run",
        "sh cuda_12.8.0_570.86.10_linux.run --silent --toolkit",
        "rm cuda_12.8.0_570.86.10_linux.run",
    )
    .pip_install(
        "torch==2.7.1+cu128",
        "torchvision==0.22.1+cu128",
        "torchaudio==2.7.1+cu128",
        "transformers>=4.57.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "flash-attn>=2.7.3",
        "triton>=3.3.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "webdataset~=0.2.33",  # zum Stream-Lesen von TAR-Shards
        "Pillow~=10.4",
        "gdown",
        "mediapipe",
        "opencv-python-headless",
        "transformers>=4.57.0",  # Qwen3-VL support
        "accelerate>=0.35.0",
        "timm>=1.0.11",
        "huggingface_hub",
        "hf_transfer",
        "numpy",
        "Pillow",
        # "facexformer-pipeline",
        "visual-debugger",
        "ultralytics",
        # @67603002024efdab26bc6f70a72fbf278e300100",
        "git+https://github.com/huggingface/diffusers",
        # "torchao >= 0.10",
        # "deepspeed>=0.15.4",
        # "bitsandbytes>=0.46.1",
        # "tiktoken",
        "scipy",
        "lpips",
        "peft",
        "git+https://github.com/xhinker/sd_embed.git@main",
        "pytest",
        "tensorboard",
        "sentencepiece",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HUB_CACHE": str(HF_CACHE),
            "TORCH_HOME": str(HF_CACHE / "torch"),
            "YOLO_CONFIG_DIR": str(HF_CACHE / "ultralytics"),
            "TF_CPP_MIN_LOG_LEVEL": "2",  # suppress TF INFO logs
            # reduce CUDA memory fragmentation
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "BNB_CUDA_VERSION": "128",
            "TORCHINDUCTOR_CACHE_DIR": str(HF_CACHE / "torchinductor"),
            # "TRANSFORMERS_VERBOSITY":"info"
        }
    )
    .add_local_dir("./configs", remote_path="/root/configs")
    .add_local_dir("./tests", remote_path="/root/tests")
    .add_local_file("./pyproject.toml", remote_path="/root/pyproject.toml")
)

hf_cache_vol = modal.Volume.from_name("cida-hf-cache", create_if_missing=True)

dataset_shard_0_vol = modal.Volume.from_name("cida-dataset-0", create_if_missing=True)
dataset_shard_1_vol = modal.Volume.from_name("cida-dataset-1", create_if_missing=True)
dataset_shard_2_vol = modal.Volume.from_name("cida-dataset-2", create_if_missing=True)
dataset_shard_3_vol = modal.Volume.from_name("cida-dataset-3", create_if_missing=True)
dataset_shard_4_vol = modal.Volume.from_name("cida-dataset-4", create_if_missing=True)
dataset_shard_5_vol = modal.Volume.from_name("cida-da4taset-5", create_if_missing=True)
dataset_shard_6_vol = modal.Volume.from_name("cida-dataset-6", create_if_missing=True)
dataset_shard_7_vol = modal.Volume.from_name("cida-dataset-7", create_if_missing=True)
dataset_shard_8_vol = modal.Volume.from_name("cida-dataset-8", create_if_missing=True)
dataset_shard_9_vol = modal.Volume.from_name("cida-dataset-9", create_if_missing=True)

preprocessed_shard_0_vol = modal.Volume.from_name(
    "cida-preprocessed-0", create_if_missing=True
)
preprocessed_shard_1_vol = modal.Volume.from_name(
    "cida-preprocessed-1", create_if_missing=True
)
preprocessed_shard_2_vol = modal.Volume.from_name(
    "cida-preprocessed-2", create_if_missing=True
)
preprocessed_shard_3_vol = modal.Volume.from_name(
    "cida-preprocessed-3", create_if_missing=True
)
preprocessed_shard_4_vol = modal.Volume.from_name(
    "cida-preprocessed-4", create_if_missing=True
)
preprocessed_shard_5_vol = modal.Volume.from_name(
    "cida-preprocessed-5", create_if_missing=True
)
preprocessed_shard_6_vol = modal.Volume.from_name(
    "cida-preprocessed-6", create_if_missing=True
)
preprocessed_shard_7_vol = modal.Volume.from_name(
    "cida-preprocessed-7", create_if_missing=True
)
preprocessed_shard_8_vol = modal.Volume.from_name(
    "cida-preprocessed-8", create_if_missing=True
)
preprocessed_shard_9_vol = modal.Volume.from_name(
    "cida-preprocessed-9", create_if_missing=True
)


output_vol = modal.Volume.from_name("cida-output", create_if_missing=True)

tensorboard_vol = modal.Volume.from_name("cida-tensorboard", create_if_missing=True)

cache_vol = modal.Volume.from_name("cida-cache", create_if_missing=True)

app = modal.App("cida", image=image)

# unsloth/Meta-Llama-3.1-8B-Instruct"
TEXT_ENCODER_4 = "meta-llama/Llama-3.1-8B-Instruct"


def models_download():
    print("Downloading models...")

    huggingface_token: str = os.environ["HUGGINGFACE_TOKEN"]

    # hf_hub_download(
    #     repo_id="lllyasviel/flux1_dev",
    #     filename="flux1-dev-fp8.safetensors",
    #     local_dir="/models/unet",  # Replace with your target directory
    #     token=huggingface_token
    # )
