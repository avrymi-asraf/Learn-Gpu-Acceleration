import torch
import time
from tools import model_size, tensor_size
# torch.cuda.memory._record_memory_history()
# Load the pre-trained WaveGlow model
waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow',weights_only=True)
waveglow.eval()  # Set to evaluation mode

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print model size
size_in_mb, total_params = model_size(waveglow)
print(f"Model size: {size_in_mb:.4f} MB, Total parameters: {total_params:,}")

# Memory before moving model to GPU
print(f"Memory allocated before moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Measure time to move model to GPU
start_time = time.time()
waveglow = waveglow.to(device)
end_time = time.time()
print(f"Time to move model to GPU: {end_time - start_time:.4f} seconds")
print(f"Memory allocated after moving model to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Define batch size and create random batch of mel-spectrograms
batch_size = 10
n_mel_channels = 80  # Number of mel channels used in WaveGlow
time_steps = 620     # Number of time steps in the spectrogram
input_shape = (batch_size, n_mel_channels, time_steps)
mel = torch.randn(input_shape,device=device)

# Print input tensor size
print(f"Input tensor size: {tensor_size(mel):.4f} MB")
print(f"Memory allocated before moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")

# Move input tensor to GPU
print(f"Memory allocated after moving input tensor to GPU: {torch.cuda.memory_allocated() / (1024 ** 2):.4f} MB")
print(f"cuda info: {torch.cuda.mem_get_info()}")

# Perform inference and measure time
with torch.no_grad():
    start_time = time.time()
    audio = waveglow.infer(mel)
    end_time = time.time()
print(f"Time to perform forward pass on GPU: {end_time - start_time:.4f} seconds, batch size: {batch_size}")

# Output information
print(f"Output audio tensor shape: {audio.shape}")
print(f"Number of audio samples generated per batch: {audio.numel()}")

# Clean up GPU memory
del waveglow, mel, audio
torch.cuda.empty_cache()
torch.cuda.memory._dump_snapshot("my_snapshot.pickle")