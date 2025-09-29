import torch


def canUseGPU() -> str:
    """
    Check if a GPU is available and can be used by PyTorch.

    Args:
        - None

    Returns:
        - str: "cuda" if a GPU is available and usable, otherwise "cpu".
    """
    if torch.cuda.is_available():
        try:
            # Create small tensors directly on the GPU
            a = torch.randn((100, 100), device="cuda")
            b = torch.randn((100, 100), device="cuda")

            # Run a computation (matrix multiplication)
            c = torch.matmul(a, b)

            # Force synchronization to trigger any CUDA errors
            torch.cuda.synchronize()
            return "cuda"
        except Exception:
            return "cpu"

    return "cpu"


if __name__ == "__main__":
    print(canUseGPU())
