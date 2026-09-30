using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace ZeroCompute.Core.Cpu
{
    /// <summary>
    /// Sovereign thread affinity and CPU core pinning manager.
    /// Eliminates thread hopping, context switching, and cache thrashing across worker threads.
    /// Pure sovereign implementation without third-party dependencies.
    /// </summary>
    internal static class CpuAffinity
    {
        private static readonly bool IsWindows = Environment.OSVersion.Platform == PlatformID.Win32NT;

        [DllImport("kernel32.dll", ExactSpelling = true)]
        private static extern IntPtr GetCurrentThread();

        [DllImport("kernel32.dll", ExactSpelling = true)]
        private static extern UIntPtr SetThreadAffinityMask(IntPtr hThread, UIntPtr dwThreadAffinityMask);

        /// <summary>
        /// Pins the calling thread strictly to a specific logical CPU core index.
        /// </summary>
        /// <param name="coreIndex">Zero-based core index (0 to ProcessorCount - 1).</param>
        /// <returns>True if successfully pinned; false otherwise.</returns>
        public static bool PinCurrentThread(int coreIndex)
        {
            if (!IsWindows || coreIndex < 0 || coreIndex >= 64)
                return false;

            try
            {
                ulong mask = 1UL << (coreIndex % 64);
                IntPtr hThread = GetCurrentThread();
                UIntPtr prev = SetThreadAffinityMask(hThread, (UIntPtr)mask);
                return prev != UIntPtr.Zero;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Sets high scheduling priority for dedicated compute worker threads.
        /// </summary>
        public static void SetHighPriority()
        {
            try
            {
                Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
            }
            catch
            {
                // Ignored in constrained execution environments
            }
        }

        private static int _physicalCoreCount;

        /// <summary>
        /// Gets the detected physical CPU core count (excluding SMT / Hyper-Threading logical threads).
        /// </summary>
        public static int PhysicalCoreCount
        {
            get
            {
                if (_physicalCoreCount > 0)
                    return _physicalCoreCount;

                int count = DetectPhysicalCores();
                _physicalCoreCount = count > 0 ? count : Math.Max(1, Environment.ProcessorCount / 2);
                return _physicalCoreCount;
            }
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool GetLogicalProcessorInformation(IntPtr buffer, ref uint returnedLength);

        [StructLayout(LayoutKind.Sequential)]
        private struct SYSTEM_LOGICAL_PROCESSOR_INFORMATION
        {
            public UIntPtr ProcessorMask;
            public int Relationship; // 0 = RelationProcessorCore
            public ulong Dummy1;
            public ulong Dummy2;
        }

        private static int DetectPhysicalCores()
        {
            if (!IsWindows)
                return Environment.ProcessorCount;

            try
            {
                uint length = 0;
                GetLogicalProcessorInformation(IntPtr.Zero, ref length);
                if (length == 0) return 0;

                IntPtr buffer = Marshal.AllocHGlobal((int)length);
                try
                {
                    if (GetLogicalProcessorInformation(buffer, ref length))
                    {
                        int structSize = Marshal.SizeOf(typeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
                        int count = (int)(length / structSize);
                        int physicalCores = 0;
                        for (int i = 0; i < count; i++)
                        {
                            IntPtr ptr = new IntPtr(buffer.ToInt64() + i * structSize);
                            var info = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION)Marshal.PtrToStructure(ptr, typeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION))!;
                            if (info.Relationship == 0) // RelationProcessorCore
                            {
                                physicalCores++;
                            }
                        }
                        if (physicalCores > 0)
                            return physicalCores;
                    }
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }
            }
            catch
            {
                // Fallback
            }

            return Math.Max(1, Environment.ProcessorCount / 2);
        }
    }
}
