using System;
using System.Runtime.InteropServices;
using ZeroCompute.Core.Blas;
using ZeroCompute.Core.Context;
using ZeroTensor.Core;

namespace ZeroCompute.Core.DirectX
{
    /// <summary>
    /// Direct3D 11 hardware compute shader dispatcher and execution context.
    /// Manages GPU devices, StructuredBuffers, and dispatches compute shaders with seamless CPU fallback.
    /// </summary>
    public sealed class D3D11ComputeContext : IComputeContext
    {
        private IntPtr _device;
        private IntPtr _context;
        private bool _isHardwareAccelerated;
        private bool _disposed;

        public ComputeBackend Backend => ComputeBackend.Direct3D11;
        public bool IsHardwareAccelerated => _isHardwareAccelerated;
        public IntPtr DeviceHandle => _device;
        public IntPtr ContextHandle => _context;

        public D3D11ComputeContext()
        {
            InitializeDevice();
        }

        private void InitializeDevice()
        {
            // Only attempt D3D11 initialization on Windows NT platforms
            bool isWindows = false;
#if NET8_0_OR_GREATER
            isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
#else
            isWindows = Environment.OSVersion.Platform == PlatformID.Win32NT;
#endif

            if (!isWindows)
            {
                _isHardwareAccelerated = false;
                return;
            }

            try
            {
                // Try Hardware first
                int hr = D3D11Native.D3D11CreateDevice(
                    IntPtr.Zero,
                    D3D11Native.D3D_DRIVER_TYPE_HARDWARE,
                    IntPtr.Zero,
                    0,
                    null,
                    0,
                    D3D11Native.D3D11_SDK_VERSION,
                    out _device,
                    out _,
                    out _context);

                if (hr < 0 || _device == IntPtr.Zero)
                {
                    // Fallback to WARP (software GPU acceleration)
                    hr = D3D11Native.D3D11CreateDevice(
                        IntPtr.Zero,
                        D3D11Native.D3D_DRIVER_TYPE_WARP,
                        IntPtr.Zero,
                        0,
                        null,
                        0,
                        D3D11Native.D3D11_SDK_VERSION,
                        out _device,
                        out _,
                        out _context);
                }

                _isHardwareAccelerated = (hr >= 0 && _device != IntPtr.Zero);
            }
            catch
            {
                _isHardwareAccelerated = false;
                _device = IntPtr.Zero;
                _context = IntPtr.Zero;
            }
        }

        public D3D11ComputeBuffer CreateStructuredBuffer<T>(int count, bool allowUav = true, bool cpuRead = false) where T : unmanaged
        {
            if (!_isHardwareAccelerated)
                throw new InvalidOperationException("Direct3D 11 device is not available on this platform/configuration.");

            int stride = Marshal.SizeOf(typeof(T));
            return new D3D11ComputeBuffer(_device, _context, count, stride, allowUav, true, cpuRead);
        }

        public IntPtr CreateComputeShader(byte[] bytecode)
        {
            if (!_isHardwareAccelerated)
                throw new InvalidOperationException("Direct3D 11 device is not available.");

            int hr = D3D11Native.CreateComputeShader(_device, bytecode, out IntPtr shader);
            if (hr < 0 || shader == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create Direct3D 11 Compute Shader (HRESULT 0x{hr:X8}).");

            return shader;
        }

        public void DispatchShader(IntPtr computeShader, int groupsX, int groupsY, int groupsZ, D3D11ComputeBuffer[] uavs, D3D11ComputeBuffer[]? srvs = null)
        {
            if (!_isHardwareAccelerated)
                throw new InvalidOperationException("Direct3D 11 device is not available.");
            if (computeShader == IntPtr.Zero)
                throw new ArgumentNullException(nameof(computeShader));

            // Set CS Shader
            D3D11Native.CSSetShader(_context, computeShader);

            // Bind UAVs
            if (uavs != null && uavs.Length > 0)
            {
                var uavHandles = new IntPtr[uavs.Length];
                for (int i = 0; i < uavs.Length; i++) uavHandles[i] = uavs[i].UavHandle;
                D3D11Native.CSSetUnorderedAccessViews(_context, 0, uavHandles);
            }

            // Bind SRVs
            if (srvs != null && srvs.Length > 0)
            {
                var srvHandles = new IntPtr[srvs.Length];
                for (int i = 0; i < srvs.Length; i++) srvHandles[i] = srvs[i].SrvHandle;
                D3D11Native.CSSetShaderResources(_context, 0, srvHandles);
            }

            // Dispatch
            D3D11Native.Dispatch(_context, (uint)groupsX, (uint)groupsY, (uint)groupsZ);

            // Unbind
            if (uavs != null && uavs.Length > 0)
            {
                var nullUavs = new IntPtr[uavs.Length];
                D3D11Native.CSSetUnorderedAccessViews(_context, 0, nullUavs);
            }
            if (srvs != null && srvs.Length > 0)
            {
                var nullSrvs = new IntPtr[srvs.Length];
                D3D11Native.CSSetShaderResources(_context, 0, nullSrvs);
            }
        }

        private IntPtr _gemmShader;
        private IntPtr _vectorShader;
        private IntPtr _activationShader;

        private IntPtr _gemmCb;
        private IntPtr _vectorCb;
        private IntPtr _activationCb;

        [StructLayout(LayoutKind.Sequential, Size = 32)]
        private struct GemmCbData
        {
            public uint M;
            public uint K;
            public uint N;
            public float Alpha;
            public float Beta;
            public float Pad0;
            public float Pad1;
            public float Pad2;
        }

        [StructLayout(LayoutKind.Sequential, Size = 16)]
        private struct VectorCbData
        {
            public uint Count;
            public uint OpType;
            public float Pad0;
            public float Pad1;
        }

        [StructLayout(LayoutKind.Sequential, Size = 16)]
        private struct ActivationCbData
        {
            public uint Count;
            public uint ActType;
            public float Pad0;
            public float Pad1;
        }

        private IntPtr CreateConstantBuffer(uint byteSize)
        {
            uint alignedSize = (byteSize + 15) & ~15u;
            var desc = new D3D11Native.D3D11_BUFFER_DESC
            {
                ByteWidth = alignedSize,
                Usage = D3D11Native.D3D11_USAGE_DEFAULT,
                BindFlags = D3D11Native.D3D11_BIND_CONSTANT_BUFFER,
                CPUAccessFlags = 0,
                MiscFlags = 0,
                StructureByteStride = 0
            };

            int hr = D3D11Native.CreateBuffer(_device, ref desc, IntPtr.Zero, out IntPtr buffer);
            if (hr < 0 || buffer == IntPtr.Zero)
                throw new InvalidOperationException($"Failed to create D3D11 Constant Buffer (0x{hr:X8})");
            return buffer;
        }

        private void EnsureGemmPipeline()
        {
            if (_gemmShader == IntPtr.Zero)
            {
                byte[] bytecode = D3D11Compiler.CompileComputeShader(D3D11Shaders.GemmShaderSource, "CSGemm");
                _gemmShader = CreateComputeShader(bytecode);
                _gemmCb = CreateConstantBuffer((uint)Marshal.SizeOf<GemmCbData>());
            }
        }

        private void EnsureVectorPipeline()
        {
            if (_vectorShader == IntPtr.Zero)
            {
                byte[] bytecode = D3D11Compiler.CompileComputeShader(D3D11Shaders.VectorOpShaderSource, "CSVectorOp");
                _vectorShader = CreateComputeShader(bytecode);
                _vectorCb = CreateConstantBuffer((uint)Marshal.SizeOf<VectorCbData>());
            }
        }

        private void EnsureActivationPipeline()
        {
            if (_activationShader == IntPtr.Zero)
            {
                byte[] bytecode = D3D11Compiler.CompileComputeShader(D3D11Shaders.ActivationShaderSource, "CSActivation");
                _activationShader = CreateComputeShader(bytecode);
                _activationCb = CreateConstantBuffer((uint)Marshal.SizeOf<ActivationCbData>());
            }
        }

        public Tensor<float> Gemm(Tensor<float> A, Tensor<float> B, float alpha = 1, float beta = 0)
        {
            if (A == null) throw new ArgumentNullException(nameof(A));
            if (B == null) throw new ArgumentNullException(nameof(B));
            if (A.Rank != 2 || B.Rank != 2)
                throw new ArgumentException("Tensors must be 2D matrices.");
            if (A.Shape[1] != B.Shape[0])
                throw new ArgumentException($"Inner dimensions must match: A is [{A.Shape[0]},{A.Shape[1]}], B is [{B.Shape[0]},{B.Shape[1]}].");

            var C = Tensor.Zeros<float>(A.Shape[0], B.Shape[1]);
            Gemm(A, B, C, alpha, beta);
            return C;
        }

        public unsafe void Gemm(Tensor<float> A, Tensor<float> B, Tensor<float> C, float alpha = 1, float beta = 0)
        {
            if (!_isHardwareAccelerated)
            {
                BlasEngine.Gemm(A, B, C, alpha, beta);
                return;
            }

            try
            {
                EnsureGemmPipeline();

                int M = A.Shape[0];
                int K = A.Shape[1];
                int N = B.Shape[1];

                var aContig = A.ToContiguous();
                var bContig = B.ToContiguous();
                var cContig = C.ToContiguous();

                using var bufA = CreateStructuredBuffer<float>(M * K, allowUav: false, cpuRead: false);
                using var bufB = CreateStructuredBuffer<float>(K * N, allowUav: false, cpuRead: false);
                using var bufC = CreateStructuredBuffer<float>(M * N, allowUav: true, cpuRead: true);

                float[] aData = new float[M * K];
                float[] bData = new float[K * N];

                aContig.AsSpan().CopyTo(aData);
                bContig.AsSpan().CopyTo(bData);

                bufA.Upload(aData);
                bufB.Upload(bData);

                if (beta != 0.0f)
                {
                    float[] cData = new float[M * N];
                    cContig.AsSpan().CopyTo(cData);
                    bufC.Upload(cData);
                }

                var cbData = new GemmCbData
                {
                    M = (uint)M,
                    K = (uint)K,
                    N = (uint)N,
                    Alpha = alpha,
                    Beta = beta
                };
                D3D11Native.UpdateSubresource(_context, _gemmCb, (IntPtr)(&cbData));
                D3D11Native.CSSetConstantBuffers(_context, 0, _gemmCb);

                int groupsX = (N + 15) / 16;
                int groupsY = (M + 15) / 16;
                DispatchShader(_gemmShader, groupsX, groupsY, 1, new[] { bufC }, new[] { bufA, bufB });

                float[] result = new float[M * N];
                bufC.Download(result);
                new Span<float>(result).CopyTo(cContig.AsSpan());

                if (!ReferenceEquals(cContig, C))
                {
                    cContig.CopyTo(C);
                }
            }
            catch
            {
                // Fallback to CPU BLAS if GPU compilation or dispatch fails
                BlasEngine.Gemm(A, B, C, alpha, beta);
            }
        }

        public unsafe void Add(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            if (!_isHardwareAccelerated)
            {
                BlasEngine.Add(A, B, C);
                return;
            }

            try
            {
                EnsureVectorPipeline();

                int count = (int)A.Length;
                var aContig = A.ToContiguous();
                var bContig = B.ToContiguous();
                var cContig = C.ToContiguous();

                using var bufA = CreateStructuredBuffer<float>(count, allowUav: false, cpuRead: false);
                using var bufB = CreateStructuredBuffer<float>(count, allowUav: false, cpuRead: false);
                using var bufC = CreateStructuredBuffer<float>(count, allowUav: true, cpuRead: true);

                float[] aData = new float[count];
                float[] bData = new float[count];
                aContig.AsSpan().CopyTo(aData);
                bContig.AsSpan().CopyTo(bData);

                bufA.Upload(aData);
                bufB.Upload(bData);

                var cbData = new VectorCbData { Count = (uint)count, OpType = 0 };
                D3D11Native.UpdateSubresource(_context, _vectorCb, (IntPtr)(&cbData));
                D3D11Native.CSSetConstantBuffers(_context, 0, _vectorCb);

                int groups = (count + 63) / 64;
                DispatchShader(_vectorShader, groups, 1, 1, new[] { bufC }, new[] { bufA, bufB });

                float[] result = new float[count];
                bufC.Download(result);
                new Span<float>(result).CopyTo(cContig.AsSpan());

                if (!ReferenceEquals(cContig, C))
                {
                    cContig.CopyTo(C);
                }
            }
            catch
            {
                BlasEngine.Add(A, B, C);
            }
        }

        public unsafe void Multiply(Tensor<float> A, Tensor<float> B, Tensor<float> C)
        {
            if (!_isHardwareAccelerated)
            {
                BlasEngine.Multiply(A, B, C);
                return;
            }

            try
            {
                EnsureVectorPipeline();

                int count = (int)A.Length;
                var aContig = A.ToContiguous();
                var bContig = B.ToContiguous();
                var cContig = C.ToContiguous();

                using var bufA = CreateStructuredBuffer<float>(count, allowUav: false, cpuRead: false);
                using var bufB = CreateStructuredBuffer<float>(count, allowUav: false, cpuRead: false);
                using var bufC = CreateStructuredBuffer<float>(count, allowUav: true, cpuRead: true);

                float[] aData = new float[count];
                float[] bData = new float[count];
                aContig.AsSpan().CopyTo(aData);
                bContig.AsSpan().CopyTo(bData);

                bufA.Upload(aData);
                bufB.Upload(bData);

                var cbData = new VectorCbData { Count = (uint)count, OpType = 1 };
                D3D11Native.UpdateSubresource(_context, _vectorCb, (IntPtr)(&cbData));
                D3D11Native.CSSetConstantBuffers(_context, 0, _vectorCb);

                int groups = (count + 63) / 64;
                DispatchShader(_vectorShader, groups, 1, 1, new[] { bufC }, new[] { bufA, bufB });

                float[] result = new float[count];
                bufC.Download(result);
                new Span<float>(result).CopyTo(cContig.AsSpan());

                if (!ReferenceEquals(cContig, C))
                {
                    cContig.CopyTo(C);
                }
            }
            catch
            {
                BlasEngine.Multiply(A, B, C);
            }
        }

        public unsafe void Activation(Tensor<float> input, Tensor<float> output, ComputeActivationType type)
        {
            if (!_isHardwareAccelerated || type == ComputeActivationType.Softmax)
            {
                BlasEngine.Activation(input, output, type);
                return;
            }

            try
            {
                EnsureActivationPipeline();

                int count = (int)input.Length;
                var inContig = input.ToContiguous();
                var outContig = output.ToContiguous();

                using var bufIn = CreateStructuredBuffer<float>(count, allowUav: false, cpuRead: false);
                using var bufOut = CreateStructuredBuffer<float>(count, allowUav: true, cpuRead: true);

                float[] inData = new float[count];
                inContig.AsSpan().CopyTo(inData);
                bufIn.Upload(inData);

                var cbData = new ActivationCbData { Count = (uint)count, ActType = (uint)type };
                D3D11Native.UpdateSubresource(_context, _activationCb, (IntPtr)(&cbData));
                D3D11Native.CSSetConstantBuffers(_context, 0, _activationCb);

                int groups = (count + 63) / 64;
                DispatchShader(_activationShader, groups, 1, 1, new[] { bufOut }, new[] { bufIn });

                float[] result = new float[count];
                bufOut.Download(result);
                new Span<float>(result).CopyTo(outContig.AsSpan());

                if (!ReferenceEquals(outContig, output))
                {
                    outContig.CopyTo(output);
                }
            }
            catch
            {
                BlasEngine.Activation(input, output, type);
            }
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
            if (!_disposed)
            {
                if (_gemmShader != IntPtr.Zero) { D3D11Native.Release(_gemmShader); _gemmShader = IntPtr.Zero; }
                if (_vectorShader != IntPtr.Zero) { D3D11Native.Release(_vectorShader); _vectorShader = IntPtr.Zero; }
                if (_activationShader != IntPtr.Zero) { D3D11Native.Release(_activationShader); _activationShader = IntPtr.Zero; }
                if (_gemmCb != IntPtr.Zero) { D3D11Native.Release(_gemmCb); _gemmCb = IntPtr.Zero; }
                if (_vectorCb != IntPtr.Zero) { D3D11Native.Release(_vectorCb); _vectorCb = IntPtr.Zero; }
                if (_activationCb != IntPtr.Zero) { D3D11Native.Release(_activationCb); _activationCb = IntPtr.Zero; }

                if (_context != IntPtr.Zero)
                {
                    D3D11Native.Release(_context);
                    _context = IntPtr.Zero;
                }
                if (_device != IntPtr.Zero)
                {
                    D3D11Native.Release(_device);
                    _device = IntPtr.Zero;
                }
                _disposed = true;
            }
        }
    }
}
