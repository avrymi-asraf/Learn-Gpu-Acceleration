import torch
from torchvision import models
import time
from tools import model_size, tensor_size

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
model.eval()  # Set to evaluation mode

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print model size
model_mem_size, total_params = model_size(model)
print(f"Model size: {model_mem_size:.4f} MB")
print(f"Total parameters: {total_params}")

# Memory before moving model to GPU
print(f"Memory allocated before moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Measure time to move model to GPU
start_time = time.time()
model = model.to(device)
end_time = time.time()
print(f"Time to move model to GPU: {end_time - start_time:.4f} seconds")
print(f"Memory allocated after moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Define batch size and create random batch of images
batch_size = 128
input_shape = (batch_size, 3, 224, 224)
batch_t = torch.randn(input_shape)

# Print input tensor size
print(f"Input tensor size: {tensor_size(batch_t):.4f} MB")
print(f"Memory allocated before moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Move input tensor to GPU
batch_t = batch_t.to(device)
print(f"Memory allocated after moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Perform inference and measure time
with torch.no_grad():
    start_time = time.time()
    output = model(batch_t)
    end_time = time.time()
print(f"Time to perform forward pass on GPU: {end_time - start_time:.4f} seconds, batch size: {batch_size}")

# Get predictions for each image in the batch
_, predicted = torch.max(output, 1)
print('Predicted classes for batch:', predicted.tolist())
print(f"Number of elements processed: {batch_size}")

# Clean up GPU memory
del model, batch_t, output
torch.cuda.empty_cache()
