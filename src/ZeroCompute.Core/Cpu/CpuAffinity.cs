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
    }
}
