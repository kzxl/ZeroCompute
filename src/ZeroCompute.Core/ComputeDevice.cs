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

        public static ComputeDevice Cpu => _defaultCpu.Value;

        public ComputeBackend Backend { get; }

        public ComputeDevice(ComputeBackend backend = ComputeBackend.CpuParallel)
        {
            Backend = backend;
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
            BlasEngine.Gemm(A, B, C, alpha, beta);
        }

        public void Add(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            BlasEngine.Add(A, B, C);
        }

        public void Multiply(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            BlasEngine.Multiply(A, B, C);
        }

        public void Activation(Tensor<float> input, Tensor<float> output, ComputeActivationType type)
        {
            BlasEngine.Activation(input, output, type);
        }

        public Tensor<float> ReduceSum(Tensor<float> input, int axis)
        {
            return BlasEngine.ReduceSum(input, axis);
        }

        public Tensor<float> ReduceMax(Tensor<float> input, int axis)
        {
            return BlasEngine.ReduceMax(input, axis);
        }

        public void Dispose()
        {
            // Resource cleanup if hardware context was allocated
        }
    }
}
