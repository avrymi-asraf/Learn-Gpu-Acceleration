import torch
import torchvision.models as models
import time

def model_size(model):
    # Calculate the size of the model in MB and the number of parameters
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    size_mb = param_size / (1024 ** 2)
    num_params = sum(p.numel() for p in model.parameters())
    return size_mb, num_params

def tensor_size(tensor):
    # Calculate the size of a tensor in MB
    return tensor.nelement() * tensor.element_size() / (1024 ** 2)

# Define a list of ready-made models from torchvision
model_names = [
'resnet18',
'alexnet',
'vgg16',
'squeezenet1_0',
'densenet161',
'inception_v3',
'googlenet',
'shufflenet_v2_x1_0',
'mobilenet_v2',
'resnext50_32x4d',
'wide_resnet50_2',
'mnasnet1_0'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate over each model, initialize it, and run a dummy input through it
for model_name in model_names:
    print(f"Running model: {model_name}")
    model = getattr(models, model_name)(pretrained=True)
    model.eval()
    
    # Calculate and display model size and number of parameters
    size_mb, num_params = model_size(model)
    print(f"Model size: {size_mb:.4f} MB, Number of parameters: {num_params:,}")
    
    # Display memory allocated before moving model to GPU
    print(f"Memory allocated before moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
    
    # Measure time to move model to GPU
    start_time = time.time()
    model_gpu = model.to(device)
    end_time = time.time()
    print(f"Time to move model to GPU: {end_time - start_time:.4f} seconds")
    print(f"Memory allocated after moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
    
    # Create random input tensor (batch size of 1, 3 color channels, 224x224 image)
    inputs_cpu = torch.randn(1024, 3, 224, 224)
    print(f"Input tensor size: {tensor_size(inputs_cpu):.4f} MB")
    print(f"Memory allocated before moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
    
    # Move input tensor to GPU
    inputs_gpu = inputs_cpu.to(device)
    print(f"Memory allocated after moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
    
    # Measure time for forward pass on GPU
    with torch.no_grad():
        start_time = time.time()
        outputs_gpu = model_gpu(inputs_gpu)
        end_time = time.time()
    print(f"Time to perform forward pass on GPU: {end_time - start_time:.4f} seconds")
    print(f"Memory allocated after forward pass: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB\n")
    
    # Clear CUDA memory
    del model_gpu, inputs_gpu, outputs_gpu
    torch.cuda.empty_cache()
    print(f"Memory allocated after clearing CUDA memory: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB\n")
