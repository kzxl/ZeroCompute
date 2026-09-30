using System;
using System.Diagnostics;
using System.Threading.Tasks;
using ZeroCompute.Core.Cpu;

namespace ZeroCompute.Benchmarks
{
    public static class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("================================================================================");
            Console.WriteLine("          ZEROCOMPUTE - CPU PARALLEL COMPUTE RUNTIME BENCHMARK SUITE             ");
            Console.WriteLine("================================================================================");
            Console.WriteLine($"Logical Processors : {Environment.ProcessorCount}");
            Console.WriteLine($"64-bit Process     : {Environment.Is64BitProcess}");
            Console.WriteLine($"OS Version         : {Environment.OSVersion}");
            Console.WriteLine($"SIMD Accelerated   : {ComputeVectorOps.IsHardwareAccelerated}");
            Console.WriteLine("================================================================================\n");

            // Warm up runtime and thread pool
            Warmup();

            RunBenchmark1_MemoryBoundArrayAdd();
            RunBenchmark2_ComputeBoundTranscendental();
            RunBenchmark3_CacheTiling2D();
            RunBenchmark4_KernelFusion();
            RunBenchmark5_ReductionSum();
            RunBenchmark6_DispatchLatencyMicro();

            Console.WriteLine("\n[DONE] Benchmark suite finished successfully.");
        }

        private static void Warmup()
        {
            float[] a = new float[1000];
            float[] b = new float[1000];
            float[] c = new float[1000];
            Compute.Vector.Add(a, b, c);
            Compute.For(1000, i => c[i] = a[i] + b[i]);
        }

        #region Benchmark 1: Memory-Bound Array Add

        private static void RunBenchmark1_MemoryBoundArrayAdd()
        {
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine("BENCHMARK 1: Array Add (Memory-Bound Workload: Low Arithmetic Intensity)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            int[] sizes = { 1_000, 100_000, 10_000_000 };
            string[] labels = { "Small (1K)", "Medium (100K)", "Huge (10M)" };

            for (int s = 0; s < sizes.Length; s++)
            {
                int n = sizes[s];
                float[] a = new float[n];
                float[] b = new float[n];
                float[] resSeq = new float[n];
                float[] resPar = new float[n];
                float[] resVec = new float[n];

                for (int i = 0; i < n; i++)
                {
                    a[i] = i * 0.1f;
                    b[i] = (i + 1) * 0.2f;
                }

                int iters = (n <= 100_000) ? 100 : 10;

                // 1. Sequential
                var sw = Stopwatch.StartNew();
                for (int iter = 0; iter < iters; iter++)
                {
                    for (int i = 0; i < n; i++) resSeq[i] = a[i] + b[i];
                }
                sw.Stop();
                double tSeq = sw.Elapsed.TotalMilliseconds / iters;

                // 2. Parallel.For
                sw.Restart();
                for (int iter = 0; iter < iters; iter++)
                {
                    Parallel.For(0, n, i => resPar[i] = a[i] + b[i]);
                }
                sw.Stop();
                double tPar = sw.Elapsed.TotalMilliseconds / iters;

                // 3. Compute.Vector.Add (Thread + SIMD unrolling + memory-aware worker cap)
                sw.Restart();
                for (int iter = 0; iter < iters; iter++)
                {
                    Compute.Vector.Add(a, b, resVec);
                }
                sw.Stop();
                double tVec = sw.Elapsed.TotalMilliseconds / iters;

                Console.WriteLine($"[{labels[s]} - N = {n:N0}]");
                Console.WriteLine($"  * Sequential         : {tSeq,8:F3} ms (1.00x)");
                Console.WriteLine($"  * Parallel.For       : {tPar,8:F3} ms ({(tSeq / tPar),5:F2}x)");
                Console.WriteLine($"  * Compute.Vector.Add : {tVec,8:F3} ms ({(tSeq / tVec),5:F2}x speedup vs Seq, {(tPar / tVec),5:F2}x vs Par.For)");
            }
            Console.WriteLine();
        }

        #endregion

        #region Benchmark 2: Compute-Bound Transcendental Math (GELU)

        private static void RunBenchmark2_ComputeBoundTranscendental()
        {
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine("BENCHMARK 2: GELU Activation (Compute-Bound: High FLOP/Byte Intensity)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            const int N = 2_000_000;
            float[] input = new float[N];
            float[] outputSeq = new float[N];
            float[] outputPar = new float[N];
            float[] outputCompute = new float[N];

            for (int i = 0; i < N; i++) input[i] = (i % 100) * 0.05f - 2.5f;

            const int iters = 5;
            const float sqrt2OverPi = 0.79788456f;

            // 1. Sequential
            var sw = Stopwatch.StartNew();
            for (int it = 0; it < iters; it++)
            {
                for (int i = 0; i < N; i++)
                {
                    float x = input[i];
                    float inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
                    outputSeq[i] = 0.5f * x * (1.0f + (float)Math.Tanh(inner));
                }
            }
            sw.Stop();
            double tSeq = sw.Elapsed.TotalMilliseconds / iters;

            // 2. Parallel.For
            sw.Restart();
            for (int it = 0; it < iters; it++)
            {
                Parallel.For(0, N, i =>
                {
                    float x = input[i];
                    float inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
                    outputPar[i] = 0.5f * x * (1.0f + (float)Math.Tanh(inner));
                });
            }
            sw.Stop();
            double tPar = sw.Elapsed.TotalMilliseconds / iters;

            // 3. Compute.For (ComputeBound worker pool with SMT-mitigated physical cores)
            sw.Restart();
            for (int it = 0; it < iters; it++)
            {
                Compute.For(N, (start, end) =>
                {
                    for (int i = start; i < end; i++)
                    {
                        float x = input[i];
                        float inner = sqrt2OverPi * (x + 0.044715f * x * x * x);
                        outputCompute[i] = 0.5f * x * (1.0f + (float)Math.Tanh(inner));
                    }
                }, CpuWorkloadType.ComputeBound);
            }
            sw.Stop();
            double tCompute = sw.Elapsed.TotalMilliseconds / iters;

            // 4. Compute.Vector.Gelu (AVX2 SIMD Vectorized Rational Approximation)
            float[] outputGelu = new float[N];
            sw.Restart();
            for (int it = 0; it < iters; it++)
            {
                Compute.Vector.Gelu(input, outputGelu);
            }
            sw.Stop();
            double tGelu = sw.Elapsed.TotalMilliseconds / iters;

            Console.WriteLine($"[N = {N:N0} elements - 5 runs averaged]");
            Console.WriteLine($"  * Sequential (Scalar) : {tSeq,8:F2} ms (1.00x)");
            Console.WriteLine($"  * Parallel.For (.NET) : {tPar,8:F2} ms ({(tSeq / tPar),5:F2}x)");
            Console.WriteLine($"  * Compute.For (Scalar): {tCompute,8:F2} ms ({(tSeq / tCompute),5:F2}x)");
            Console.WriteLine($"  * Compute.Vector.Gelu : {tGelu,8:F2} ms ({(tSeq / tGelu),5:F2}x speedup vs Seq, {(tPar / tGelu),5:F2}x vs Par.For!)");
            Console.WriteLine();
        }

        #endregion

        #region Benchmark 3: Matrix Multiplication (Cache Tiling vs Naive)

        private static void RunBenchmark3_CacheTiling2D()
        {
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine("BENCHMARK 3: Matrix Multiplication (Cache Tiling vs Naive Stride-N Scanning)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            const int M = 512;
            const int K = 512;
            const int N = 512;

            var tA = ZeroTensor.Core.Tensor.Zeros<float>(M, K);
            var tB = ZeroTensor.Core.Tensor.Zeros<float>(K, N);
            var tC_Naive = ZeroTensor.Core.Tensor.Zeros<float>(M, N);
            var tC_Tiled = ZeroTensor.Core.Tensor.Zeros<float>(M, N);

            for (int i = 0; i < M; i++)
                for (int k = 0; k < K; k++)
                    tA[i, k] = (i + k) * 0.01f;

            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++)
                    tB[k, j] = (k + j) * 0.01f;

            // 1. Naive row-by-row Parallel.For without cache tiling
            var sw = Stopwatch.StartNew();
            Parallel.For(0, M, i =>
            {
                for (int k = 0; k < K; k++)
                {
                    float aVal = tA[i, k];
                    for (int j = 0; j < N; j++)
                    {
                        tC_Naive[i, j] += aVal * tB[k, j];
                    }
                }
            });
            sw.Stop();
            double tNaive = sw.Elapsed.TotalMilliseconds;

            // 2. Cache-Tiled GEMM using ZeroCompute BlasEngine (64x64 tiles + Compute.For + unrolling)
            sw.Restart();
            ZeroCompute.Core.Blas.BlasEngine.Gemm(tA, tB, tC_Tiled);
            sw.Stop();
            double tTiled = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine($"[Matrix Size: {M} x {K} x {N} ({M * N * K:N0} operations)]");
            Console.WriteLine($"  * Naive Parallel.For : {tNaive,8:F2} ms (1.00x)");
            Console.WriteLine($"  * Cache-Tiled GEMM   : {tTiled,8:F2} ms ({(tNaive / tTiled),5:F2}x speedup via L1/L2 Cache Tiling)");
            Console.WriteLine();
        }

        #endregion

        #region Benchmark 4: Kernel Fusion

        private static void RunBenchmark4_KernelFusion()
        {
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine("BENCHMARK 4: Kernel Fusion (Single-pass L1 Cache vs 3-pass Memory Access)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            const int N = 2_000_000;
            float[] src = new float[N];
            float[] temp1 = new float[N];
            float[] temp2 = new float[N];
            float[] dstMulti = new float[N];
            float[] dstFused = new float[N];

            for (int i = 0; i < N; i++) src[i] = i * 0.05f - 50_000f;

            const int iters = 10;

            // 1. Multi-pass (3 round-trips to DRAM)
            var sw = Stopwatch.StartNew();
            for (int it = 0; it < iters; it++)
            {
                // Pass 1: Multiply 2.0
                Parallel.For(0, N, i => temp1[i] = src[i] * 2.0f);
                // Pass 2: Add 10.0
                Parallel.For(0, N, i => temp2[i] = temp1[i] + 10.0f);
                // Pass 3: ReLU
                Parallel.For(0, N, i => dstMulti[i] = temp2[i] > 0f ? temp2[i] : 0f);
            }
            sw.Stop();
            double tMulti = sw.Elapsed.TotalMilliseconds / iters;

            // 2. Fused (1 round-trip to DRAM, intermediate in L1/Registers)
            sw.Restart();
            for (int it = 0; it < iters; it++)
            {
                Compute.FusedMap(src, dstFused, x =>
                {
                    float v = x * 2.0f + 10.0f;
                    return v > 0f ? v : 0f;
                });
            }
            sw.Stop();
            double tFused = sw.Elapsed.TotalMilliseconds / iters;

            Console.WriteLine($"[N = {N:N0} elements - 10 runs averaged]");
            Console.WriteLine($"  * Multi-Pass (3x DRAM) : {tMulti,8:F2} ms (1.00x)");
            Console.WriteLine($"  * Compute.FusedMap     : {tFused,8:F2} ms ({(tMulti / tFused),5:F2}x speedup - 66% DRAM bandwidth saved)");
            Console.WriteLine();
        }

        #endregion

        #region Benchmark 5: Parallel Reduction

        private static void RunBenchmark5_ReductionSum()
        {
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine("BENCHMARK 5: Parallel Reduction (Sum across 10 Million elements)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            const int N = 10_000_000;
            float[] data = new float[N];
            for (int i = 0; i < N; i++) data[i] = (i % 100) * 0.1f;

            const int iters = 10;

            // 1. Scalar loop
            var sw = Stopwatch.StartNew();
            float sumSeq = 0f;
            for (int it = 0; it < iters; it++)
            {
                sumSeq = 0f;
                for (int i = 0; i < N; i++) sumSeq += data[i];
            }
            sw.Stop();
            double tSeq = sw.Elapsed.TotalMilliseconds / iters;

            // 2. Compute.ReduceSum (Multi-core + 4-way SIMD accumulator unrolling)
            sw.Restart();
            float sumCompute = 0f;
            for (int it = 0; it < iters; it++)
            {
                sumCompute = Compute.ReduceSum(data);
            }
            sw.Stop();
            double tCompute = sw.Elapsed.TotalMilliseconds / iters;

            Console.WriteLine($"[N = {N:N0} elements - 10 runs averaged]");
            Console.WriteLine($"  * Scalar Loop      : {tSeq,8:F2} ms (1.00x)");
            Console.WriteLine($"  * Compute.ReduceSum: {tCompute,8:F2} ms ({(tSeq / tCompute),5:F2}x speedup)");
            Console.WriteLine();
        }

        #endregion

        #region Benchmark 6: Dispatch Latency Micro-benchmark

        private static void RunBenchmark6_DispatchLatencyMicro()
        {
            Console.WriteLine("--------------------------------------------------------------------------------");
            Console.WriteLine("BENCHMARK 6: Dispatch Overhead (50,000 rapid small-N loops, N = 100)");
            Console.WriteLine("--------------------------------------------------------------------------------");

            const int N = 100;
            const int iters = 50_000;
            int[] data = new int[N];

            // 1. Parallel.For
            var sw = Stopwatch.StartNew();
            for (int it = 0; it < iters; it++)
            {
                Parallel.For(0, N, i => { data[i] = i; });
            }
            sw.Stop();
            double tPar = sw.Elapsed.TotalMilliseconds;

            // 2. Compute.For (Deterministic sequential cutoff avoids thread dispatch)
            sw.Restart();
            for (int it = 0; it < iters; it++)
            {
                Compute.For(N, i => { data[i] = i; });
            }
            sw.Stop();
            double tCompute = sw.Elapsed.TotalMilliseconds;

            Console.WriteLine($"[50,000 iterations of N=100 loop]");
            Console.WriteLine($"  * Parallel.For : {tPar,8:F1} ms total ({tPar / iters * 1000,6:F2} µs/loop)");
            Console.WriteLine($"  * Compute.For  : {tCompute,8:F1} ms total ({tCompute / iters * 1000,6:F2} µs/loop, {(tPar / tCompute),5:F1}x faster via Planner cutoff)");
            Console.WriteLine();
        }

        #endregion
    }
}
