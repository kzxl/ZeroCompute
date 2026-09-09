using System;
using ZeroCompute.Core.Context;
using ZeroTensor.Core;

namespace ZeroCompute.Core.Context
{
    /// <summary>
    /// Unified compute interface for accelerating tensor operations on CPU multi-threading or GPU hardware.
    /// </summary>
    public interface IComputeContext : IDisposable
    {
        ComputeBackend Backend { get; }

        /// <summary>
        /// General Matrix Multiplication: C = alpha * (A x B) + beta * C.
        /// </summary>
        void Gemm(
            Tensor<float> A,
            Tensor<float> B,
            Tensor<float> C,
            float alpha = 1.0f,
            float beta = 0.0f);

        /// <summary>
        /// Element-wise addition: C = A + B.
        /// </summary>
        void Add(Tensor<float> A, Tensor<float> B, Tensor<float> C);

        /// <summary>
        /// Element-wise multiplication: C = A * B.
        /// </summary>
        void Multiply(Tensor<float> A, Tensor<float> B, Tensor<float> C);

        /// <summary>
        /// Applies element-wise activation function.
        /// </summary>
        void Activation(Tensor<float> input, Tensor<float> output, ComputeActivationType type);

        /// <summary>
        /// Reduces tensor along the specified axis (e.g., sum, max, mean).
        /// </summary>
        Tensor<float> ReduceSum(Tensor<float> input, int axis);
        Tensor<float> ReduceMax(Tensor<float> input, int axis);
    }
}
