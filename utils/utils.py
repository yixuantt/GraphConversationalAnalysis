import torch


def prepare_device(keep_reproducibility=False):
    if keep_reproducibility:
        print("Using CuDNN deterministic mode in the experiment.")
        torch.backends.cudnn.benchmark = False  # ensures that CUDA selects the same convolution algorithm each time
        torch.set_deterministic(True)  # configures PyTorch only to use deterministic implementation
    else:
        torch.backends.cudnn.benchmark = True
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
