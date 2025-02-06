import torch
from torchvision import models
import time
from tools import model_size, tensor_size

# Load the pre-trained MobileNetV2 model
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.eval()  # Set to evaluation mode

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print model size
size_in_mb, total_params = model_size(model)
print(f"Model size: {size_in_mb:.4f} MB, Total parameters: {total_params}")

# Memory before moving model to GPU
if device.type == 'cuda':
    print(f"Memory allocated before moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Measure time to move model to GPU
start_time = time.time()
model = model.to(device)
end_time = time.time()
print(f"Time to move model to {device.type.upper()}: {end_time - start_time:.4f} seconds")
if device.type == 'cuda':
    print(f"Memory allocated after moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Define batch size and create random batch of images
batch_size = 128
input_shape = (batch_size, 3, 224, 224)
batch_t = torch.randn(input_shape)

# Print input tensor size and number of elements
input_size_in_mb = tensor_size(batch_t)
print(f"Input tensor size: {input_size_in_mb:.4f} MB")
print(f"Number of elements in input batch: {batch_t.numel()}")
if device.type == 'cuda':
    print(f"Memory allocated before moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Move input tensor to GPU
batch_t = batch_t.to(device)
if device.type == 'cuda':
    print(f"Memory allocated after moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Perform inference and measure time
with torch.no_grad():
    start_time = time.time()
    output = model(batch_t)
    end_time = time.time()
print(f"Time to perform forward pass on {device.type.upper()}: {end_time - start_time:.4f} seconds, batch size: {batch_size}")

# Get predictions for each image in the batch
_, predicted = torch.max(output, 1)
print('Predicted classes for batch:', predicted.tolist())

# Clean up GPU memory
del model, batch_t, output
if device.type == 'cuda':
    torch.cuda.empty_cache()
