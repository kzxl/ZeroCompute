using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace ZeroCompute.Core.Cpu
{
    /// <summary>
    /// Vectorized SIMD operations using hardware acceleration (SSE, AVX, AVX2, AVX-512)
    /// via <see cref="Vector{T}"/> with 4-way loop unrolling for maximum instruction-level parallelism (ILP).
    /// Pure C# unsafe blocks compile directly to unaligned vector instructions on all supported runtimes.
    /// </summary>
    public static unsafe class ComputeVectorOps
    {
        private static readonly int VectorSize = Vector<float>.Count;

        /// <summary>
        /// Gets whether hardware SIMD vector acceleration is active on the current machine.
        /// </summary>
        public static bool IsHardwareAccelerated => Vector.IsHardwareAccelerated;

        /// <summary>
        /// Element-wise addition: result[i] = a[i] + b[i].
        /// </summary>
        public static void Add(float* pA, float* pB, float* pResult, int start, int end)
        {
            int count = end - start;
            if (count <= 0) return;

            int i = start;
            int vStep = VectorSize;
            int unrollStep = vStep * 4;

            if (Vector.IsHardwareAccelerated && count >= unrollStep)
            {
                int unrollEnd = start + (count / unrollStep) * unrollStep;
                for (; i < unrollEnd; i += unrollStep)
                {
                    var vA0 = *(Vector<float>*)(pA + i);
                    var vB0 = *(Vector<float>*)(pB + i);
                    var vA1 = *(Vector<float>*)(pA + i + vStep);
                    var vB1 = *(Vector<float>*)(pB + i + vStep);
                    var vA2 = *(Vector<float>*)(pA + i + vStep * 2);
                    var vB2 = *(Vector<float>*)(pB + i + vStep * 2);
                    var vA3 = *(Vector<float>*)(pA + i + vStep * 3);
                    var vB3 = *(Vector<float>*)(pB + i + vStep * 3);

                    *(Vector<float>*)(pResult + i) = vA0 + vB0;
                    *(Vector<float>*)(pResult + i + vStep) = vA1 + vB1;
                    *(Vector<float>*)(pResult + i + vStep * 2) = vA2 + vB2;
                    *(Vector<float>*)(pResult + i + vStep * 3) = vA3 + vB3;
                }
            }

            if (Vector.IsHardwareAccelerated && i <= end - vStep)
            {
                int singleEnd = start + (count / vStep) * vStep;
                for (; i < singleEnd; i += vStep)
                {
                    *(Vector<float>*)(pResult + i) = *(Vector<float>*)(pA + i) + *(Vector<float>*)(pB + i);
                }
            }

            // Scalar tail
            for (; i < end; i++)
            {
                pResult[i] = pA[i] + pB[i];
            }
        }

        /// <summary>
        /// Element-wise multiplication: result[i] = a[i] * b[i].
        /// </summary>
        public static void Multiply(float* pA, float* pB, float* pResult, int start, int end)
        {
            int count = end - start;
            if (count <= 0) return;

            int i = start;
            int vStep = VectorSize;
            int unrollStep = vStep * 4;

            if (Vector.IsHardwareAccelerated && count >= unrollStep)
            {
                int unrollEnd = start + (count / unrollStep) * unrollStep;
                for (; i < unrollEnd; i += unrollStep)
                {
                    var vA0 = *(Vector<float>*)(pA + i);
                    var vB0 = *(Vector<float>*)(pB + i);
                    var vA1 = *(Vector<float>*)(pA + i + vStep);
                    var vB1 = *(Vector<float>*)(pB + i + vStep);
                    var vA2 = *(Vector<float>*)(pA + i + vStep * 2);
                    var vB2 = *(Vector<float>*)(pB + i + vStep * 2);
                    var vA3 = *(Vector<float>*)(pA + i + vStep * 3);
                    var vB3 = *(Vector<float>*)(pB + i + vStep * 3);

                    *(Vector<float>*)(pResult + i) = vA0 * vB0;
                    *(Vector<float>*)(pResult + i + vStep) = vA1 * vB1;
                    *(Vector<float>*)(pResult + i + vStep * 2) = vA2 * vB2;
                    *(Vector<float>*)(pResult + i + vStep * 3) = vA3 * vB3;
                }
            }

            if (Vector.IsHardwareAccelerated && i <= end - vStep)
            {
                int singleEnd = start + (count / vStep) * vStep;
                for (; i < singleEnd; i += vStep)
                {
                    *(Vector<float>*)(pResult + i) = *(Vector<float>*)(pA + i) * *(Vector<float>*)(pB + i);
                }
            }

            for (; i < end; i++)
            {
                pResult[i] = pA[i] * pB[i];
            }
        }

        /// <summary>
        /// Element-wise scale: result[i] = a[i] * scalar.
        /// </summary>
        public static void Scale(float* pA, float scalar, float* pResult, int start, int end)
        {
            int count = end - start;
            if (count <= 0) return;

            int i = start;
            int vStep = VectorSize;
            int unrollStep = vStep * 4;
            var vScalar = new Vector<float>(scalar);

            if (Vector.IsHardwareAccelerated && count >= unrollStep)
            {
                int unrollEnd = start + (count / unrollStep) * unrollStep;
                for (; i < unrollEnd; i += unrollStep)
                {
                    *(Vector<float>*)(pResult + i) = *(Vector<float>*)(pA + i) * vScalar;
                    *(Vector<float>*)(pResult + i + vStep) = *(Vector<float>*)(pA + i + vStep) * vScalar;
                    *(Vector<float>*)(pResult + i + vStep * 2) = *(Vector<float>*)(pA + i + vStep * 2) * vScalar;
                    *(Vector<float>*)(pResult + i + vStep * 3) = *(Vector<float>*)(pA + i + vStep * 3) * vScalar;
                }
            }

            if (Vector.IsHardwareAccelerated && i <= end - vStep)
            {
                int singleEnd = start + (count / vStep) * vStep;
                for (; i < singleEnd; i += vStep)
                {
                    *(Vector<float>*)(pResult + i) = *(Vector<float>*)(pA + i) * vScalar;
                }
            }

            for (; i < end; i++)
            {
                pResult[i] = pA[i] * scalar;
            }
        }

        /// <summary>
        /// Fused Multiply-Add: result[i] = a[i] * b[i] + c[i].
        /// </summary>
        public static void Fma(float* pA, float* pB, float* pC, float* pResult, int start, int end)
        {
            int count = end - start;
            if (count <= 0) return;

            int i = start;
            int vStep = VectorSize;
            int unrollStep = vStep * 4;

            if (Vector.IsHardwareAccelerated && count >= unrollStep)
            {
                int unrollEnd = start + (count / unrollStep) * unrollStep;
                for (; i < unrollEnd; i += unrollStep)
                {
                    var vA0 = *(Vector<float>*)(pA + i);
                    var vB0 = *(Vector<float>*)(pB + i);
                    var vC0 = *(Vector<float>*)(pC + i);

                    var vA1 = *(Vector<float>*)(pA + i + vStep);
                    var vB1 = *(Vector<float>*)(pB + i + vStep);
                    var vC1 = *(Vector<float>*)(pC + i + vStep);

                    var vA2 = *(Vector<float>*)(pA + i + vStep * 2);
                    var vB2 = *(Vector<float>*)(pB + i + vStep * 2);
                    var vC2 = *(Vector<float>*)(pC + i + vStep * 2);

                    var vA3 = *(Vector<float>*)(pA + i + vStep * 3);
                    var vB3 = *(Vector<float>*)(pB + i + vStep * 3);
                    var vC3 = *(Vector<float>*)(pC + i + vStep * 3);

                    *(Vector<float>*)(pResult + i) = (vA0 * vB0) + vC0;
                    *(Vector<float>*)(pResult + i + vStep) = (vA1 * vB1) + vC1;
                    *(Vector<float>*)(pResult + i + vStep * 2) = (vA2 * vB2) + vC2;
                    *(Vector<float>*)(pResult + i + vStep * 3) = (vA3 * vB3) + vC3;
                }
            }

            if (Vector.IsHardwareAccelerated && i <= end - vStep)
            {
                int singleEnd = start + (count / vStep) * vStep;
                for (; i < singleEnd; i += vStep)
                {
                    *(Vector<float>*)(pResult + i) = (*(Vector<float>*)(pA + i) * *(Vector<float>*)(pB + i)) + *(Vector<float>*)(pC + i);
                }
            }

            for (; i < end; i++)
            {
                pResult[i] = (pA[i] * pB[i]) + pC[i];
            }
        }
    }
}
