using System;
using System.Threading.Tasks;
using ZeroCompute.Core.Context;
using ZeroTensor.Core;

namespace ZeroCompute.Core.Blas
{
    /// <summary>
    /// High-performance CPU multi-threaded BLAS engine with cache-blocked GEMM and vectorized element-wise kernels.
    /// Pure C# with zero third-party dependencies.
    /// </summary>
    public static class BlasEngine
    {
        private const int BlockSize = 64;

        /// <summary>
        /// Cache-blocked multi-threaded Matrix Multiplication: C = alpha * (A x B) + beta * C.
        /// A is [M, K], B is [K, N], C is [M, N].
        /// </summary>
        public static unsafe void Gemm(
            Tensor<float> A,
            Tensor<float> B,
            Tensor<float> C,
            float alpha = 1.0f,
            float beta = 0.0f)
        {
            if (A == null) throw new ArgumentNullException(nameof(A));
            if (B == null) throw new ArgumentNullException(nameof(B));
            if (C == null) throw new ArgumentNullException(nameof(C));

            if (A.Rank != 2 || B.Rank != 2 || C.Rank != 2)
                throw new ArgumentException("Tensors must be 2D matrices.");

            int M = A.Shape[0];
            int K = A.Shape[1];
            int N = B.Shape[1];

            if (B.Shape[0] != K)
                throw new ArgumentException($"Inner dimensions must match: A is [{M},{K}], B is [{B.Shape[0]},{N}].");
            if (C.Shape[0] != M || C.Shape[1] != N)
                throw new ArgumentException($"Output matrix C shape must be [{M},{N}], but got [{C.Shape[0]},{C.Shape[1]}].");

            var aContig = A.ToContiguous();
            var bContig = B.ToContiguous();
            var cContig = C.ToContiguous();

            fixed (float* pA = &aContig[0, 0])
            fixed (float* pB = &bContig[0, 0])
            fixed (float* pC = &cContig[0, 0])
            {
                IntPtr ptrA = (IntPtr)pA;
                IntPtr ptrB = (IntPtr)pB;
                IntPtr ptrC = (IntPtr)pC;

                int totalElements = M * N;

                // Handle beta scaling first if needed
                if (beta == 0.0f)
                {
                    Parallel.For(0, (totalElements + 1023) / 1024, chunk =>
                    {
                        float* localC = (float*)ptrC;
                        int start = chunk * 1024;
                        int end = Math.Min(start + 1024, totalElements);
                        for (int idx = start; idx < end; idx++)
                        {
                            localC[idx] = 0.0f;
                        }
                    });
                }
                else if (beta != 1.0f)
                {
                    Parallel.For(0, (totalElements + 1023) / 1024, chunk =>
                    {
                        float* localC = (float*)ptrC;
                        int start = chunk * 1024;
                        int end = Math.Min(start + 1024, totalElements);
                        for (int idx = start; idx < end; idx++)
                        {
                            localC[idx] *= beta;
                        }
                    });
                }

                // Tiled blocked GEMM
                int numBlocksM = (M + BlockSize - 1) / BlockSize;

                Parallel.For(0, numBlocksM, bi =>
                {
                    float* localA = (float*)ptrA;
                    float* localB = (float*)ptrB;
                    float* localC = (float*)ptrC;

                    int iStart = bi * BlockSize;
                    int iEnd = Math.Min(iStart + BlockSize, M);

                    for (int bk = 0; bk < K; bk += BlockSize)
                    {
                        int kEnd = Math.Min(bk + BlockSize, K);

                        for (int bj = 0; bj < N; bj += BlockSize)
                        {
                            int jEnd = Math.Min(bj + BlockSize, N);

                            // Inner kernel
                            for (int i = iStart; i < iEnd; i++)
                            {
                                float* pRowC = localC + i * N;
                                for (int k = bk; k < kEnd; k++)
                                {
                                    float aVal = alpha * localA[i * K + k];
                                    float* pRowB = localB + k * N;

                                    int j = bj;
                                    // 4-way loop unrolling
                                    for (; j <= jEnd - 4; j += 4)
                                    {
                                        pRowC[j] += aVal * pRowB[j];
                                        pRowC[j + 1] += aVal * pRowB[j + 1];
                                        pRowC[j + 2] += aVal * pRowB[j + 2];
                                        pRowC[j + 3] += aVal * pRowB[j + 3];
                                    }
                                    for (; j < jEnd; j++)
                                    {
                                        pRowC[j] += aVal * pRowB[j];
                                    }
                                }
                            }
                        }
                    }
                });
            }

            if (!ReferenceEquals(cContig, C))
            {
                cContig.CopyTo(C);
            }
        }

        public static unsafe void Add(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            int total = (int)A.Length;
            var aContig = A.ToContiguous();
            var bContig = B.ToContiguous();
            var cContig = C.ToContiguous();

            var aFlat = aContig.Flatten();
            var bFlat = bContig.Flatten();
            var cFlat = cContig.Flatten();

            fixed (float* pA = &aFlat[0])
            fixed (float* pB = &bFlat[0])
            fixed (float* pC = &cFlat[0])
            {
                IntPtr ptrA = (IntPtr)pA;
                IntPtr ptrB = (IntPtr)pB;
                IntPtr ptrC = (IntPtr)pC;

                Parallel.For(0, (total + 2047) / 2048, chunk =>
                {
                    float* localA = (float*)ptrA;
                    float* localB = (float*)ptrB;
                    float* localC = (float*)ptrC;

                    int start = chunk * 2048;
                    int end = Math.Min(start + 2048, total);
                    for (int i = start; i < end; i++)
                    {
                        localC[i] = localA[i] + localB[i];
                    }
                });
            }

            if (!ReferenceEquals(cContig, C))
            {
                cContig.CopyTo(C);
            }
        }

        public static unsafe void Multiply(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            int total = (int)A.Length;
            var aContig = A.ToContiguous();
            var bContig = B.ToContiguous();
            var cContig = C.ToContiguous();

            var aFlat = aContig.Flatten();
            var bFlat = bContig.Flatten();
            var cFlat = cContig.Flatten();

            fixed (float* pA = &aFlat[0])
            fixed (float* pB = &bFlat[0])
            fixed (float* pC = &cFlat[0])
            {
                IntPtr ptrA = (IntPtr)pA;
                IntPtr ptrB = (IntPtr)pB;
                IntPtr ptrC = (IntPtr)pC;

                Parallel.For(0, (total + 2047) / 2048, chunk =>
                {
                    float* localA = (float*)ptrA;
                    float* localB = (float*)ptrB;
                    float* localC = (float*)ptrC;

                    int start = chunk * 2048;
                    int end = Math.Min(start + 2048, total);
                    for (int i = start; i < end; i++)
                    {
                        localC[i] = localA[i] * localB[i];
                    }
                });
            }

            if (!ReferenceEquals(cContig, C))
            {
                cContig.CopyTo(C);
            }
        }

        public static unsafe void Activation(Tensor<float> input, Tensor<float> output, ComputeActivationType type)
        {
            int total = (int)input.Length;
            var inContig = input.ToContiguous();
            var outContig = output.ToContiguous();

            var inFlat = inContig.Flatten();
            var outFlat = outContig.Flatten();

            fixed (float* pIn = &inFlat[0])
            fixed (float* pOut = &outFlat[0])
            {
                IntPtr ptrIn = (IntPtr)pIn;
                IntPtr ptrOut = (IntPtr)pOut;

                switch (type)
                {
                    case ComputeActivationType.ReLU:
                        Parallel.For(0, (total + 2047) / 2048, chunk =>
                        {
                            float* localIn = (float*)ptrIn;
                            float* localOut = (float*)ptrOut;
                            int start = chunk * 2048;
                            int end = Math.Min(start + 2048, total);
                            for (int i = start; i < end; i++)
                            {
                                float val = localIn[i];
                                localOut[i] = val > 0f ? val : 0f;
                            }
                        });
                        break;

                    case ComputeActivationType.LeakyReLU:
                        Parallel.For(0, (total + 2047) / 2048, chunk =>
                        {
                            float* localIn = (float*)ptrIn;
                            float* localOut = (float*)ptrOut;
                            int start = chunk * 2048;
                            int end = Math.Min(start + 2048, total);
                            for (int i = start; i < end; i++)
                            {
                                float val = localIn[i];
                                localOut[i] = val >= 0f ? val : 0.01f * val;
                            }
                        });
                        break;

                    case ComputeActivationType.Sigmoid:
                        Parallel.For(0, (total + 2047) / 2048, chunk =>
                        {
                            float* localIn = (float*)ptrIn;
                            float* localOut = (float*)ptrOut;
                            int start = chunk * 2048;
                            int end = Math.Min(start + 2048, total);
                            for (int i = start; i < end; i++)
                            {
                                localOut[i] = 1.0f / (1.0f + (float)Math.Exp(-localIn[i]));
                            }
                        });
                        break;

                    case ComputeActivationType.Tanh:
                        Parallel.For(0, (total + 2047) / 2048, chunk =>
                        {
                            float* localIn = (float*)ptrIn;
                            float* localOut = (float*)ptrOut;
                            int start = chunk * 2048;
                            int end = Math.Min(start + 2048, total);
                            for (int i = start; i < end; i++)
                            {
                                localOut[i] = (float)Math.Tanh(localIn[i]);
                            }
                        });
                        break;

                    case ComputeActivationType.GELU:
                        const float sqrt2OverPi = 0.79788456f;
                        Parallel.For(0, (total + 2047) / 2048, chunk =>
                        {
                            float* localIn = (float*)ptrIn;
                            float* localOut = (float*)ptrOut;
                            int start = chunk * 2048;
                            int end = Math.Min(start + 2048, total);
                            for (int i = start; i < end; i++)
                            {
                                float x = localIn[i];
                                float inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
                                localOut[i] = 0.5f * x * (1.0f + (float)Math.Tanh(inner));
                            }
                        });
                        break;

                    case ComputeActivationType.Softmax:
                        int rows = (int)(input.Length / input.Shape[input.Rank - 1]);
                        int cols = input.Shape[input.Rank - 1];

                        Parallel.For(0, rows, r =>
                        {
                            float* localIn = (float*)ptrIn;
                            float* localOut = (float*)ptrOut;

                            float* rowIn = localIn + r * cols;
                            float* rowOut = localOut + r * cols;

                            float max = float.MinValue;
                            for (int c = 0; c < cols; c++)
                                if (rowIn[c] > max) max = rowIn[c];

                            float sum = 0.0f;
                            for (int c = 0; c < cols; c++)
                            {
                                float exp = (float)Math.Exp(rowIn[c] - max);
                                rowOut[c] = exp;
                                sum += exp;
                            }

                            float invSum = 1.0f / sum;
                            for (int c = 0; c < cols; c++)
                            {
                                rowOut[c] *= invSum;
                            }
                        });
                        break;
                }
            }

            if (!ReferenceEquals(outContig, output))
            {
                outContig.CopyTo(output);
            }
        }

        public static unsafe Tensor<float> ReduceSum(Tensor<float> input, int axis)
        {
            if (axis < 0) axis += input.Rank;
            if (axis < 0 || axis >= input.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            int[] newShape = new int[input.Rank - 1];
            int idx = 0;
            for (int i = 0; i < input.Rank; i++)
            {
                if (i != axis) newShape[idx++] = input.Shape[i];
            }

            var output = Tensor.Zeros<float>(newShape.Length == 0 ? new[] { 1 } : newShape);

            int outer = 1;
            for (int i = 0; i < axis; i++) outer *= input.Shape[i];
            int reduceDim = input.Shape[axis];
            int inner = 1;
            for (int i = axis + 1; i < input.Rank; i++) inner *= input.Shape[i];

            var inContig = input.ToContiguous();
            var outContig = output.ToContiguous();

            var inFlat = inContig.Flatten();
            var outFlat = outContig.Flatten();

            fixed (float* pIn = &inFlat[0])
            fixed (float* pOut = &outFlat[0])
            {
                IntPtr ptrIn = (IntPtr)pIn;
                IntPtr ptrOut = (IntPtr)pOut;

                Parallel.For(0, outer, o =>
                {
                    float* localIn = (float*)ptrIn;
                    float* localOut = (float*)ptrOut;

                    for (int inr = 0; inr < inner; inr++)
                    {
                        float sum = 0f;
                        for (int k = 0; k < reduceDim; k++)
                        {
                            sum += localIn[(o * reduceDim + k) * inner + inr];
                        }
                        localOut[o * inner + inr] = sum;
                    }
                });
            }

            if (!ReferenceEquals(outContig, output))
            {
                outContig.CopyTo(output);
            }

            return output;
        }

        public static unsafe Tensor<float> ReduceMax(Tensor<float> input, int axis)
        {
            if (axis < 0) axis += input.Rank;
            if (axis < 0 || axis >= input.Rank)
                throw new ArgumentOutOfRangeException(nameof(axis));

            int[] newShape = new int[input.Rank - 1];
            int idx = 0;
            for (int i = 0; i < input.Rank; i++)
            {
                if (i != axis) newShape[idx++] = input.Shape[i];
            }

            var output = Tensor.Zeros<float>(newShape.Length == 0 ? new[] { 1 } : newShape);

            int outer = 1;
            for (int i = 0; i < axis; i++) outer *= input.Shape[i];
            int reduceDim = input.Shape[axis];
            int inner = 1;
            for (int i = axis + 1; i < input.Rank; i++) inner *= input.Shape[i];

            var inContig = input.ToContiguous();
            var outContig = output.ToContiguous();

            var inFlat = inContig.Flatten();
            var outFlat = outContig.Flatten();

            fixed (float* pIn = &inFlat[0])
            fixed (float* pOut = &outFlat[0])
            {
                IntPtr ptrIn = (IntPtr)pIn;
                IntPtr ptrOut = (IntPtr)pOut;

                Parallel.For(0, outer, o =>
                {
                    float* localIn = (float*)ptrIn;
                    float* localOut = (float*)ptrOut;

                    for (int inr = 0; inr < inner; inr++)
                    {
                        float max = float.MinValue;
                        for (int k = 0; k < reduceDim; k++)
                        {
                            float v = localIn[(o * reduceDim + k) * inner + inr];
                            if (v > max) max = v;
                        }
                        localOut[o * inner + inr] = max;
                    }
                });
            }

            if (!ReferenceEquals(outContig, output))
            {
                outContig.CopyTo(output);
            }

            return output;
        }
    }
}
