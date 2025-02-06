import torch
from transformers import BertModel
import time
from tools import model_size, tensor_size

# Load the pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased')
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

# Define batch size and create random batch of input_ids and attention_mask
batch_size = 128
seq_length = 128
vocab_size = model.config.vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

# Print input tensor size
print(f"Input tensor size: {tensor_size(input_ids):.4f} MB")
print(f"Memory allocated before moving input tensors to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Move input tensors to GPU
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)
print(f"Memory allocated after moving input tensors to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Perform inference and measure time
with torch.no_grad():
    start_time = time.time()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    end_time = time.time()
print(f"Time to perform forward pass on GPU: {end_time - start_time:.4f} seconds, batch size: {batch_size}")

# Clean up GPU memory
del model, input_ids, attention_mask, outputs
torch.cuda.empty_cache()
