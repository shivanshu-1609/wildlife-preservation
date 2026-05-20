# save this as check_cuda_cpu.py
import torch

def check_device():
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    
    # Print results
    print(f"CUDA Available: {cuda_available}")
    print(f"Device in use: {device}")
    
    # If CUDA is available, show GPU details
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Total GPUs: {torch.cuda.device_count()}")
    else:
        print("CUDA is not available. Using CPU.")

if __name__ == "__main__":
    check_device()