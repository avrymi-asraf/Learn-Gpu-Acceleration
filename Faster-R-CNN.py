import torch
from torchvision import models
import time
from tools import model_size, tensor_size

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
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
input_images = [torch.randn(3, 800, 800) for _ in range(batch_size)]  # Using 800x800 images

# Print total input tensor size and number of elements
total_input_size = sum([tensor_size(img) for img in input_images])
print(f"Total input tensor size: {total_input_size:.4f} MB")
print(f"Number of elements in input batch: {sum([img.numel() for img in input_images])}")
if device.type == 'cuda':
    print(f"Memory allocated before moving input tensors to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Move input tensors to GPU
input_images = [img.to(device) for img in input_images]
if device.type == 'cuda':
    print(f"Memory allocated after moving input tensors to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Perform inference and measure time
with torch.no_grad():
    start_time = time.time()
    outputs = model(input_images)
    end_time = time.time()
print(f"Time to perform forward pass on {device.type.upper()}: {end_time - start_time:.4f} seconds, batch size: {batch_size}")

# Print the number of detections for each image
for idx, output in enumerate(outputs):
    num_detections = len(output['boxes'])
    print(f"Image {idx} has {num_detections} detections")

# Clean up GPU memory
del model, input_images, outputs
if device.type == 'cuda':
    torch.cuda.empty_cache()
