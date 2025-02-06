# PyTorch CUDA Module Examples Using torch.cuda Functions

# Import necessary libraries
import torch
import time

# Example 1: Check GPU Availability using torch.cuda.is_available()
if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
else:
    print("CUDA is not available. Using CPU.")

# Example 2: Get the Device Name
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Device Name: {gpu_name}")

# Example 3: Get the Number of Available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

# Example 4: Set the Current Device
if torch.cuda.is_available() and num_gpus > 1:
    torch.cuda.set_device(1)  # Set to second GPU if available
    current_device = torch.cuda.current_device()
    print(f"Current Device: {current_device}, Device Name: {torch.cuda.get_device_name(current_device)}")

# Example 5: Memory Management with torch.cuda.memory_allocated() and torch.cuda.memory_cached()
if torch.cuda.is_available():
    tensor_a = torch.randn(1000, 1000, device="cuda")
    allocated_memory = torch.cuda.memory_allocated(0)
    cached_memory = torch.cuda.memory_reserved(0)
    print(f"Memory Allocated: {allocated_memory / 1e6} MB")
    print(f"Memory Reserved (Cached): {cached_memory / 1e6} MB")

# Example 6: Manual Memory Management with torch.cuda.empty_cache()
# Create a large tensor to allocate memory
if torch.cuda.is_available():
    large_tensor = torch.randn(10000, 10000, device="cuda")
    print(f"Memory Allocated After Large Tensor Creation: {torch.cuda.memory_allocated(0) / 1e6} MB")
    
    # Delete the tensor and empty the cache
    del large_tensor
    torch.cuda.empty_cache()
    print(f"Memory Allocated After Emptying Cache: {torch.cuda.memory_allocated(0) / 1e6} MB")

# Example 7: Synchronizing CUDA Operations using torch.cuda.synchronize()
# Create two random tensors and perform addition
if torch.cuda.is_available():
    tensor_b = torch.randn(5000, 5000, device="cuda")
    tensor_c = torch.randn(5000, 5000, device="cuda")
    
    start_time = time.time()
    tensor_d = tensor_b + tensor_c
    torch.cuda.synchronize()  # Ensure all operations are completed
    end_time = time.time()
    print(f"Time taken for tensor addition with synchronization: {end_time - start_time:.4f} seconds")

# Example 8: CUDA Streams for Overlapping Operations
# Create two streams and perform operations concurrently
if torch.cuda.is_available():
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    tensor_e = torch.randn(1000, 1000, device="cuda")
    tensor_f = torch.randn(1000, 1000, device="cuda")
    
    # Perform operations in different streams
    with torch.cuda.stream(stream1):
        tensor_g = tensor_e.pow(2)
    with torch.cuda.stream(stream2):
        tensor_h = tensor_f.mul(3)
    
    # Wait for all streams to complete
    torch.cuda.synchronize()
    print("Operations completed using CUDA streams.")
