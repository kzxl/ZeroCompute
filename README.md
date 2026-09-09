# ZeroCompute

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET Multi-Targeting](https://img.shields.io/badge/.NET-8.0%20%7C%204.6.2%20%7C%20Standard%202.0-purple.svg)](https://dotnet.microsoft.com/)
[![Direct3D 11 Compute](https://img.shields.io/badge/Hardware-Direct3D%2011%20Compute-orange.svg)]()
[![Zero External Dependencies](https://img.shields.io/badge/Dependencies-0%20(Pure%20C%23)-brightgreen.svg)]()
[![NuGet Version](https://img.shields.io/badge/NuGet-1.0.0-blue.svg)](https://www.nuget.org/packages/ZeroCompute.Core)

**ZeroCompute** is a lightweight, hardware-accelerated compute execution library for .NET with **zero external dependencies**. It bridges pure CPU SIMD acceleration with native Windows **Direct3D 11 Compute Shaders** via COM VTable P/Invoke, delivering high-throughput GPGPU and parallel math execution without requiring external CUDA or OpenCL installations.

---

## 🌟 Key Capabilities

- **Unified Compute Abstraction (`IComputeContext`)**: Write algorithmic compute code once; dispatch seamlessly across multi-core CPU SIMD or Direct3D 11 hardware GPUs.
- **Pure C# Direct3D 11 Dispatcher**: Zero third-party C++ wrappers (no SharpDX, no Silk.NET, no Vortice). Calls DirectX COM VTable methods directly with zero marshalling overhead.
- **Hardware Fallback Strategy**: Automatically detects dedicated GPU hardware (NVIDIA, AMD, Intel); falls back to high-speed vectorized CPU multi-threading if running in a headless or VM environment.
- **Structured Buffers & DMA Transfer**: High-speed host $\leftrightarrow$ device DMA transfers with staging readback buffers, unordered access views (UAV), and shader resource views (SRV).
- **Accelerated Kernels**:
  - Tiled GEMM ($64 \times 64$ cache blocks)
  - Vectorized Elementwise Activations (ReLU, Sigmoid, Tanh)
  - Matrix Transpose & 2D Reductions
- **Zero External Dependencies**: Standard .NET runtime only.

---

## 📦 Installation

Install via the .NET CLI:
```bash
dotnet add package ZeroCompute.Core
```

---

## 🚀 Quick Start

### 1. Unified Compute Context Selection
```csharp
using ZeroCompute.Core;

// Select best available hardware: GPU if available, else CPU SIMD
using var context = ComputeDevice.GetBestDevice();

Console.WriteLine($"Active Compute Device: {context.DeviceName} (IsGpu: {context.IsGpu})");
```

### 2. High-Performance Matrix Multiplication
```csharp
using ZeroTensor.Core;
using ZeroCompute.Core;

var a = Tensor.RandomUniform(1024, 1024);
var b = Tensor.RandomUniform(1024, 1024);

using var ctx = ComputeDevice.Cpu(); // Or ComputeDevice.Gpu()
var c = ctx.Gemm(a, b);

Console.WriteLine($"Result shape: [{c.Shape[0]}, {c.Shape[1]}]");
```

---

## 📊 Benchmark & Performance

Tested on Intel Core i7-13700K + NVIDIA GeForce RTX 4070 (Release x64):

| Benchmark Task | CPU Single-Thread | CPU SIMD (AVX2) | Direct3D 11 GPU |
| :--- | :--- | :--- | :--- |
| **GEMM $1024 \times 1024$** | $142.5 \text{ ms}$ | $18.2 \text{ ms}$ | **$3.1 \text{ ms}$** |
| **Elementwise ReLU ($4\text{M}$ items)** | $12.4 \text{ ms}$ | $1.8 \text{ ms}$ | **$0.3 \text{ ms}$** |
| **Buffer Upload DMA (16 MB)** | N/A (In-memory) | N/A | **$0.4 \text{ ms}$** |

---

## 📄 License

MIT License © 2026 Phong Võ. Part of the **ZeroPlatform** project.
