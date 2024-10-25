# Import necessary libraries
import torch
import torch.nn as nn
import time

import torch

def tensor_size(tensor):
    return (tensor.numel() * torch.finfo(tensor.dtype).bits / 8) / (1024 ** 2)

def model_size(model):
    total_size = 0
    total_params = 0
    for param in model.parameters():
        total_size += param.numel() * torch.finfo(param.dtype).bits / 8
        total_params += param.numel()
    return total_size / (1024 ** 2), total_params  # Convert size to MB

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example: Create a simple model and move it to GPU
simple_model = nn.Sequential(
    nn.Linear(3 * 32 * 32, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

# using model in gpu
simple_model_gpu = simple_model.to(device)
x = torch.rand(10, 3 * 32 * 32).to(device)
output = simple_model_gpu(x)

# Define a function to create models of different sizes
# Linear Model
def create_linear_model(num_layers, layer_size):
    layers = []
    input_size = 3 * 32 * 32
    for _ in range(num_layers):
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(nn.ReLU())
        input_size = layer_size
    layers.append(nn.Linear(input_size, 10))
    return nn.Sequential(*layers)

# Transformer Model
def create_transformer_model(num_encoder_layers, d_model, num_heads, dim_feedforward):
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
    fc = nn.Linear(d_model, 10)
    return nn.Sequential(transformer_encoder, fc)

# Define model configurations
model_configs = {
    "linear": [
        (10, 10000),
        # (20, 10000),
        # (30, 10000),
        # (40, 10000),
        # (50, 10000)
    ],
    "transformer": [
        (1, 64, 4, 256),
        (2, 128, 8, 512),
        (3, 256, 8, 1024)
    ]
}

# Measure the time taken to move each model to the GPU and perform a forward pass
for model_type, configs in model_configs.items():
    for idx, config in enumerate(configs, start=1):
        print(f"\n{model_type.capitalize()} Model {idx}: {config}")
        
        # Create model based on type
        if model_type == "linear":
            model = create_linear_model(*config)
        elif model_type == "transformer":
            model = create_transformer_model(*config)
        else:
            continue
        
        # Measure model size and number of parameters
        size_mb, num_params = model_size(model)
        print(f"Model size: {size_mb:.4f} MB, Number of parameters: {num_params:,}")
        print(f"Memory allocated before moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
        
        # Measure time to move model to GPU
        start_time = time.time()
        model_gpu = model.to(device)
        end_time = time.time()
        print(f"Time to move model to GPU: {end_time - start_time:.4f} seconds")
        print(f"Memory allocated after moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
        
        # Create random input tensor
        if model_type == "linear":
            inputs_cpu = torch.randn(1024, 3 * 32 * 32)
        elif model_type == "transformer":
            d_model = config[1]
            inputs_cpu = torch.randn(1024, d_model)  # Adjusting input size based on d_model
        else:
            continue
        
        print(f"Input tensor size: {tensor_size(inputs_cpu):.4f} MB")
        print(f"Memory allocated before moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
        
        inputs_gpu = inputs_cpu.to(device)
        print(f"Memory allocated after moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
        
        # Measure time for forward pass on GPU
        start_time = time.time()
        outputs_gpu = model_gpu(inputs_gpu)
        end_time = time.time()
        print(f"Time to perform forward pass on GPU: {end_time - start_time:.4f} seconds")
        print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
        
        # Clean up GPU memory
        del model_gpu, inputs_gpu, outputs_gpu
        torch.cuda.empty_cache()
