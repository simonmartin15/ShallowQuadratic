import torch


class ClearCache:
    """Clears CUDA cache before and after simulating"""
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
