using System;
using ZeroCompute.Core.Blas;
using ZeroCompute.Core.Context;
using ZeroTensor.Core;

namespace ZeroCompute.Core
{
    /// <summary>
    /// Factory and coordinator for high-performance hardware and CPU compute dispatch.
    /// </summary>
    public sealed class ComputeDevice : IComputeContext
    {
        private static readonly Lazy<ComputeDevice> _defaultCpu = new Lazy<ComputeDevice>(() => new ComputeDevice(ComputeBackend.CpuParallel));
        private static readonly Lazy<ComputeDevice> _defaultGpu = new Lazy<ComputeDevice>(() => new ComputeDevice(ComputeBackend.Direct3D11));

        public static ComputeDevice Cpu => _defaultCpu.Value;
        public static ComputeDevice Gpu => _defaultGpu.Value;

        private readonly ZeroCompute.Core.DirectX.D3D11ComputeContext? _d3d11Context;

        public ComputeBackend Backend { get; }
        public bool IsHardwareAccelerated => _d3d11Context?.IsHardwareAccelerated ?? false;

        public ComputeDevice(ComputeBackend backend = ComputeBackend.CpuParallel)
        {
            Backend = backend;
            if (backend == ComputeBackend.Direct3D11)
            {
                _d3d11Context = new ZeroCompute.Core.DirectX.D3D11ComputeContext();
            }
        }

        public static ComputeDevice Create(ComputeBackend backend = ComputeBackend.CpuParallel)
        {
            return new ComputeDevice(backend);
        }

        public void Gemm(
            Tensor<float> A,
            Tensor<float> B,
            Tensor<float> C,
            float alpha = 1.0f,
            float beta = 0.0f)
        {
            if (_d3d11Context != null)
                _d3d11Context.Gemm(A, B, C, alpha, beta);
            else
                BlasEngine.Gemm(A, B, C, alpha, beta);
        }

        public void Add(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            if (_d3d11Context != null)
                _d3d11Context.Add(A, B, C);
            else
                BlasEngine.Add(A, B, C);
        }

        public void Multiply(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            if (_d3d11Context != null)
                _d3d11Context.Multiply(A, B, C);
            else
                BlasEngine.Multiply(A, B, C);
        }

        public void Activation(Tensor<float> input, Tensor<float> output, ComputeActivationType type)
        {
            if (_d3d11Context != null)
                _d3d11Context.Activation(input, output, type);
            else
                BlasEngine.Activation(input, output, type);
        }

        public Tensor<float> ReduceSum(Tensor<float> input, int axis)
        {
            if (_d3d11Context != null)
                return _d3d11Context.ReduceSum(input, axis);
            return BlasEngine.ReduceSum(input, axis);
        }

        public Tensor<float> ReduceMax(Tensor<float> input, int axis)
        {
            if (_d3d11Context != null)
                return _d3d11Context.ReduceMax(input, axis);
            return BlasEngine.ReduceMax(input, axis);
        }

        public void Dispose()
        {
            _d3d11Context?.Dispose();
        }
    }
}
