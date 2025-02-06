# Learn using GPU acceleration

## Overview

This project aims to demonstrate the use of libraries that support GPU acceleration and illustrate the performance improvements they offer. The project uses Jupyter Notebooks for learning purposes.

## Goals

- Learn to use all libraries with GPU acceleration for various tasks, including data loading, processing, neural network training, and deploying models.
- Analyze runtimes for each library to determine the most efficient one in terms of GPU utilization.
- Learn the tools to care cuda, how use it, how know the memory allocated ect'

## Libraries&#x20;

The following machine learning libraries are included:

- [ ] TensorFlow
- [x] PyTorch
- [ ] scikit-learn (limited GPU support)
- [ ] XGBoost
- [ ] LightGBM
- [ ] RAPIDS (cuML)
- [ ] ONNX Runtime
- [ ] Transformers
- [ ] TensorRT
- [ ] TorchServe
- [ ] MLflow
- [ ] NVIDIA Triton Inference Server

## Using CUDA

### Core CUDA Operations
- [x] Create tensors and move them to CUDA device
- [x] Monitor and manage GPU memory usage 
- [x] Compare runtime between CPU and CUDA operations 

### Advanced CUDA Features
- [ ] Work with CUDA streams for concurrent operations
- [ ] Use CUDA events for precise timing measurements
- [ ] Implement synchronization points with `torch.cuda.synchronize()`
- [ ] Using torch.profiler.profile [https://pytorch.org/blog/understanding-gpu-memory-1/](link)

### Performance Optimization
- [x] Analyze transfer overhead between CPU and GPU
- [x] Benchmark different batch sizes for optimal throughput
- [x] Profile GPU utilization during model training

### Debugging and Monitoring
- [x] Use CUDA memory profiler for leak detection
- [ ] Monitor GPU temperature and utilization
- [ ] Implement error handling for out-of-memory scenarios

## Ready-made models from PyTorch
- [x] **ResNet-18** (Image Classification, Convolutional Neural Network)
- [x] **VGG-16** (Image Classification, Convolutional Neural Network)
- [x] **MobileNetV2** (Image Classification, Lightweight CNN)
- [ ] **Faster R-CNN** (Object Detection) - cuda out of memory
- [x] **FCN (Fully Convolutional Network)** (Semantic Segmentation)
- [x] **BERT (Bidirectional Encoder Representations from Transformers)** (Natural Language Processing)
- [x] **GPT-2** (Generative Pretrained Transformer 2) (Text Generation)
- [x] **waveglow** (Audio Generation) - only one example
- [ ] **Autoencoder** (Dimensionality Reduction)
- [ ] **LeNet** (Image Classification, Convolutional Neural Network)
- [ ] **EfficientNet** (Image Classification, Efficient CNN)
- [ ] **YOLOv5** (Object Detection)
- [ ] **Temporal Convolutional Network (TCN)** (Time Series Forecasting)
- [ ] **UNet** (Image Segmentation)
- [ ] **OpenAI CLIP** (Contrastive Language–Image Pre-training)
- [ ] **StyleGAN2** (Image Generation)

## Getting Started

### Prerequisites

To run the benchmarks, you need:

- A computer with a GPU and up-to-date drivers (NVIDIA CUDA-capable GPU recommended).
- Python (>=3.8).
- CUDA Toolkit (version compatible with your GPU).
- Python package manager (`pip` or `conda`).

### Installation

Clone the repository and install dependencies:

Ensure that the CUDA Toolkit is correctly installed and available for GPU libraries to function properly.

## Using

Each library has an accompanying Jupyter Notebook that illustrates its usage, along with exercises for practice.

## Contributions

Contributions are welcome! If you have ideas for improving the benchmark process or want to add more libraries, feel free to create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to the open-source community for their incredible machine learning tools and libraries, which make this kind of research possible.

## Contact

For any questions, please open an issue on GitHub or reach out to [your.email@example.com](mailto\:your.email@example.com).

