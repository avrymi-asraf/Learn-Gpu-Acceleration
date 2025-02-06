# Import necessary libraries
import torch

def tensor_size(tensor):
    return (tensor.nelement() * tensor.element_size()) / (1024 ** 2)  # Size in MB

def model_size(model):
    total_size = 0
    total_params = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
        total_params += param.numel()
    return total_size / (1024 ** 2), total_params  # Size in MB
