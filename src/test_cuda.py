import torch
import torchvision
import torchaudio

def check_pytorch_installation():
    # Check if PyTorch is installed
    print("Checking PyTorch installation...")
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get the current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
        print(f"Current device index: {current_device}")

        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")

def check_torchvision_torchaudio_installation():
    # Check torchvision and torchaudio installations
    print("\nChecking torchvision and torchaudio installation...")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"torchaudio version: {torchaudio.__version__}")

if __name__ == "__main__":
    check_pytorch_installation()
    check_torchvision_torchaudio_installation()
