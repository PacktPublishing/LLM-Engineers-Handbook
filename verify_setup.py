#!/usr/bin/env python3
"""
Verify Unsloth dependencies are installed correctly.
Run this script to check if your environment is ready for fine-tuning.
"""

import sys
from typing import Dict, Tuple


def check_dependency(name: str, import_path: str) -> Tuple[bool, str]:
    """Check if a dependency is installed and return version."""
    try:
        module = __import__(import_path)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)


def check_gpu_support() -> Dict[str, bool]:
    """Check available GPU backends."""
    try:
        import torch

        return {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        }
    except ImportError:
        return {"cuda": False, "mps": False, "cuda_version": None}


def main():
    print("=" * 70)
    print("🔍 Unsloth Dependencies Verification")
    print("=" * 70)
    print()

    # Core dependencies
    dependencies = {
        "PyTorch": "torch",
        "Transformers": "transformers",
        "Datasets": "datasets",
        "TRL": "trl",
        "PEFT": "peft",
        "Accelerate": "accelerate",
        "HuggingFace Hub": "huggingface_hub",
        "Python-dotenv": "dotenv",
        "Scipy": "scipy",
        "tqdm": "tqdm",
    }

    print("📦 Core Dependencies:")
    print("-" * 70)

    all_installed = True
    for name, import_path in dependencies.items():
        installed, version = check_dependency(name, import_path)
        status = "✅" if installed else "❌"
        version_str = f"v{version}" if installed else f"NOT INSTALLED ({version})"
        print(f"  {status} {name:20s} {version_str}")
        if not installed:
            all_installed = False

    print()
    print("🎮 GPU Support:")
    print("-" * 70)

    gpu_info = check_gpu_support()
    cuda_status = "✅" if gpu_info["cuda"] else "❌"
    mps_status = "✅" if gpu_info["mps"] else "❌"

    print(f"  {cuda_status} CUDA Support: {'Available' if gpu_info['cuda'] else 'Not Available'}")
    if gpu_info["cuda_version"]:
        print(f"     └─ CUDA Version: {gpu_info['cuda_version']}")
    print(f"  {mps_status} MPS Support:  {'Available' if gpu_info['mps'] else 'Not Available'}")

    print()
    print("🚀 Unsloth Library:")
    print("-" * 70)

    unsloth_installed, unsloth_version = check_dependency("Unsloth", "unsloth")
    if unsloth_installed:
        print(f"  ✅ Unsloth: v{unsloth_version}")
        print(f"  ✅ Ready for training!")
    else:
        print(f"  ❌ Unsloth: NOT INSTALLED")
        print()
        print("  ⚠️  Unsloth requires NVIDIA GPU with CUDA support")
        print()
        if gpu_info["mps"] and not gpu_info["cuda"]:
            print("  📱 Detected: macOS with Apple Silicon (MPS)")
            print("     - MPS is not compatible with Unsloth")
            print("     - Unsloth requires CUDA (NVIDIA GPUs only)")
        else:
            print("  🔧 To install Unsloth on a CUDA-enabled system:")
            print('     pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')

    print()
    print("=" * 70)
    print("📋 Summary:")
    print("=" * 70)

    if all_installed and not gpu_info["cuda"]:
        print("  ✅ All dependencies installed for local development")
        print("  ⚠️  No CUDA GPU detected - training requires cloud GPU")
        print()
        print("  🌐 Recommended for training:")
        print("     1. Google Colab (free T4 GPU)")
        print("     2. Cloud GPU providers (RunPod, Lambda Labs, Vast.ai)")
        print("     3. AWS SageMaker with GPU instances")
        print()
        print("  📓 Ready-to-use notebook: notebooks/unsloth_finetuning.ipynb")
    elif all_installed and gpu_info["cuda"] and unsloth_installed:
        print("  ✅ ALL SYSTEMS GO! Ready for local training with Unsloth!")
    elif all_installed and gpu_info["cuda"] and not unsloth_installed:
        print("  ✅ Dependencies installed")
        print("  ✅ CUDA GPU detected")
        print("  ⚠️  Install Unsloth to start training:")
        print('     pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"')
    else:
        print("  ❌ Some dependencies are missing. Please install them first.")
        print("     pip install -r llm_engineering/model/finetuning/requirements-unsloth.txt")

    print("=" * 70)
    print()

    return 0 if all_installed else 1


if __name__ == "__main__":
    sys.exit(main())
