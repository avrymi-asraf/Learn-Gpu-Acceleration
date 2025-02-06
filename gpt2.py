import torch
from transformers import GPT2Model
import time
from tools import model_size, tensor_size

# Load the pre-trained GPT-2 model
model = GPT2Model.from_pretrained('gpt2')
model.eval()  # Set to evaluation mode

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print model size
size_in_mb, _ = model_size(model)
print(f"Model size: {size_in_mb:.4f} MB")

# Memory before moving model to GPU
print(f"Memory allocated before moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Measure time to move model to GPU
start_time = time.time()
model.to(device)
end_time = time.time()
print(f"Time to move model to GPU: {end_time - start_time:.4f} seconds")
print(f"Memory allocated after moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Define batch size and create random input_ids
batch_size = 1024
seq_length = 50
vocab_size = model.config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))

# Print input tensor size
print(f"Input tensor size: {tensor_size(input_ids):.4f} MB")
print(f"Memory allocated before moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Move input tensor to GPU
input_ids = input_ids.to(device)
print(f"Memory allocated after moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Perform inference and measure time
with torch.no_grad():
    start_time = time.time()
    outputs = model(input_ids=input_ids)
    end_time = time.time()
print(f"Time to perform forward pass on GPU: {end_time - start_time:.4f} seconds, batch size: {batch_size}")

# Clean up GPU memory
del model, input_ids, outputs
torch.cuda.empty_cache()
