namespace ZeroCompute.Core.DirectX
{
    /// <summary>
    /// Embedded DirectCompute HLSL compute shaders (Shader Model 5.0) for accelerated tensor operations.
    /// Pure C# runtime string templates compiled dynamically via d3dcompiler_47.dll.
    /// </summary>
    internal static class D3D11Shaders
    {
        public const string GemmShaderSource = @"
cbuffer GemmParams : register(b0)
{
    uint M;
    uint K;
    uint N;
    float Alpha;
    float Beta;
    float3 Pad;
};

StructuredBuffer<float> MatrixA : register(t0);
StructuredBuffer<float> MatrixB : register(t1);
RWStructuredBuffer<float> MatrixC : register(u0);

#define TILE_DIM 16
groupshared float tileA[TILE_DIM][TILE_DIM];
groupshared float tileB[TILE_DIM][TILE_DIM];

[numthreads(TILE_DIM, TILE_DIM, 1)]
void CSGemm(uint3 threadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    uint row = groupId.y * TILE_DIM + threadId.y;
    uint col = groupId.x * TILE_DIM + threadId.x;
    float acc = 0.0f;

    uint numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (uint t = 0; t < numTiles; t++)
    {
        uint tiledColA = t * TILE_DIM + threadId.x;
        uint tiledRowB = t * TILE_DIM + threadId.y;

        tileA[threadId.y][threadId.x] = (row < M && tiledColA < K) ? MatrixA[row * K + tiledColA] : 0.0f;
        tileB[threadId.y][threadId.x] = (tiledRowB < K && col < N) ? MatrixB[tiledRowB * N + col] : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_DIM; k++)
        {
            acc += tileA[threadId.y][k] * tileB[k][threadId.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < N)
    {
        uint idx = row * N + col;
        if (Beta == 0.0f)
            MatrixC[idx] = Alpha * acc;
        else
            MatrixC[idx] = Alpha * acc + Beta * MatrixC[idx];
    }
}
";

        public const string BatchedGemmShaderSource = @"
cbuffer BatchedGemmParams : register(b0)
{
    uint M;
    uint K;
    uint N;
    uint BatchCount;
    float Alpha;
    float Beta;
    float2 Pad;
};

StructuredBuffer<float> BatchMatrixA : register(t0);
StructuredBuffer<float> BatchMatrixB : register(t1);
RWStructuredBuffer<float> BatchMatrixC : register(u0);

#define TILE_DIM 16
groupshared float b_tileA[TILE_DIM][TILE_DIM];
groupshared float b_tileB[TILE_DIM][TILE_DIM];

[numthreads(TILE_DIM, TILE_DIM, 1)]
void CSBatchedGemm(uint3 threadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    uint b = groupId.z;
    if (b >= BatchCount) return;

    uint row = groupId.y * TILE_DIM + threadId.y;
    uint col = groupId.x * TILE_DIM + threadId.x;
    float acc = 0.0f;

    uint offsetA = b * (M * K);
    uint offsetB = b * (K * N);
    uint offsetC = b * (M * N);

    uint numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (uint t = 0; t < numTiles; t++)
    {
        uint tiledColA = t * TILE_DIM + threadId.x;
        uint tiledRowB = t * TILE_DIM + threadId.y;

        b_tileA[threadId.y][threadId.x] = (row < M && tiledColA < K) ? BatchMatrixA[offsetA + row * K + tiledColA] : 0.0f;
        b_tileB[threadId.y][threadId.x] = (tiledRowB < K && col < N) ? BatchMatrixB[offsetB + tiledRowB * N + col] : 0.0f;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint k = 0; k < TILE_DIM; k++)
        {
            acc += b_tileA[threadId.y][k] * b_tileB[k][threadId.x];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < N)
    {
        uint idx = offsetC + row * N + col;
        if (Beta == 0.0f)
            BatchMatrixC[idx] = Alpha * acc;
        else
            BatchMatrixC[idx] = Alpha * acc + Beta * BatchMatrixC[idx];
    }
}
";

        public const string VectorOpShaderSource = @"
cbuffer VectorParams : register(b0)
{
    uint Count;
    uint OpType;
    float2 Pad;
};

StructuredBuffer<float> InputA : register(t0);
StructuredBuffer<float> InputB : register(t1);
RWStructuredBuffer<float> OutputC : register(u0);

[numthreads(64, 1, 1)]
void CSVectorOp(uint3 id : SV_DispatchThreadID)
{
    if (id.x < Count)
    {
        if (OpType == 0)
            OutputC[id.x] = InputA[id.x] + InputB[id.x];
        else
            OutputC[id.x] = InputA[id.x] * InputB[id.x];
    }
}
";

        public const string ActivationShaderSource = @"
cbuffer ActParams : register(b0)
{
    uint Count;
    uint ActType;
    float2 Pad;
};

StructuredBuffer<float> ActInput : register(t0);
RWStructuredBuffer<float> ActOutput : register(u0);

[numthreads(64, 1, 1)]
void CSActivation(uint3 id : SV_DispatchThreadID)
{
    if (id.x < Count)
    {
        float x = ActInput[id.x];
        float res = 0.0f;

        if (ActType == 0) // ReLU
        {
            res = max(0.0f, x);
        }
        else if (ActType == 1) // Sigmoid
        {
            res = 1.0f / (1.0f + exp(-x));
        }
        else if (ActType == 2) // Tanh
        {
            res = tanh(x);
        }
        else if (ActType == 3) // GELU (textbook tanh approximation)
        {
            float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
            res = 0.5f * x * (1.0f + tanh(inner));
        }
        else if (ActType == 4) // SiLU / Swish
        {
            res = x / (1.0f + exp(-x));
        }

        ActOutput[id.x] = res;
    }
}
";

        public const string RmsNormShaderSource = @"
cbuffer RmsNormParams : register(b0)
{
    uint HiddenDim;
    uint RowCount;
    float Epsilon;
    uint HasWeight;
};

StructuredBuffer<float> RmsInput : register(t0);
StructuredBuffer<float> RmsWeight : register(t1);
RWStructuredBuffer<float> RmsOutput : register(u0);

groupshared float s_rmsSum[256];

[numthreads(256, 1, 1)]
void CSRmsNorm(uint3 threadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    uint row = groupId.x;
    if (row >= RowCount) return;

    uint tid = threadId.x;
    uint rowOffset = row * HiddenDim;

    float localSqSum = 0.0f;
    for (uint i = tid; i < HiddenDim; i += 256)
    {
        float val = RmsInput[rowOffset + i];
        localSqSum += val * val;
    }
    s_rmsSum[tid] = localSqSum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_rmsSum[tid] += s_rmsSum[tid + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    float meanSq = s_rmsSum[0] / (float)HiddenDim;
    float invRms = rsqrt(meanSq + Epsilon);

    for (uint j = tid; j < HiddenDim; j += 256)
    {
        float w = (HasWeight != 0) ? RmsWeight[j] : 1.0f;
        RmsOutput[rowOffset + j] = RmsInput[rowOffset + j] * invRms * w;
    }
}
";

        public const string LayerNormShaderSource = @"
cbuffer LayerNormParams : register(b0)
{
    uint HiddenDim;
    uint RowCount;
    float Epsilon;
    uint Flags; // Bit 0: HasWeight, Bit 1: HasBias
};

StructuredBuffer<float> LnInput : register(t0);
StructuredBuffer<float> LnWeight : register(t1);
StructuredBuffer<float> LnBias : register(t2);
RWStructuredBuffer<float> LnOutput : register(u0);

groupshared float s_lnMean[256];
groupshared float s_lnVar[256];

[numthreads(256, 1, 1)]
void CSLayerNorm(uint3 threadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    uint row = groupId.x;
    if (row >= RowCount) return;

    uint tid = threadId.x;
    uint rowOffset = row * HiddenDim;

    // Pass 1: Compute Mean
    float localSum = 0.0f;
    for (uint i = tid; i < HiddenDim; i += 256)
    {
        localSum += LnInput[rowOffset + i];
    }
    s_lnMean[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s) s_lnMean[tid] += s_lnMean[tid + s];
        GroupMemoryBarrierWithGroupSync();
    }
    float mean = s_lnMean[0] / (float)HiddenDim;
    GroupMemoryBarrierWithGroupSync();

    // Pass 2: Compute Variance
    float localVar = 0.0f;
    for (uint j = tid; j < HiddenDim; j += 256)
    {
        float diff = LnInput[rowOffset + j] - mean;
        localVar += diff * diff;
    }
    s_lnVar[tid] = localVar;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint v = 128; v > 0; v >>= 1)
    {
        if (tid < v) s_lnVar[tid] += s_lnVar[tid + v];
        GroupMemoryBarrierWithGroupSync();
    }
    float variance = s_lnVar[0] / (float)HiddenDim;
    float invStd = rsqrt(variance + Epsilon);

    // Pass 3: Normalize and scale/shift
    bool hasWeight = (Flags & 1) != 0;
    bool hasBias = (Flags & 2) != 0;

    for (uint k = tid; k < HiddenDim; k += 256)
    {
        float val = (LnInput[rowOffset + k] - mean) * invStd;
        float w = hasWeight ? LnWeight[k] : 1.0f;
        float b = hasBias ? LnBias[k] : 0.0f;
        LnOutput[rowOffset + k] = val * w + b;
    }
}
";

        public const string SoftmaxShaderSource = @"
cbuffer SoftmaxParams : register(b0)
{
    uint RowLength;
    uint RowCount;
    float2 Pad;
};

StructuredBuffer<float> SmInput : register(t0);
RWStructuredBuffer<float> SmOutput : register(u0);

groupshared float s_smMax[256];
groupshared float s_smSum[256];

[numthreads(256, 1, 1)]
void CSSoftmax(uint3 threadId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    uint row = groupId.x;
    if (row >= RowCount) return;

    uint tid = threadId.x;
    uint rowOffset = row * RowLength;

    // Pass 1: Find Max
    float localMax = -3.402823466e+38F;
    for (uint i = tid; i < RowLength; i += 256)
    {
        localMax = max(localMax, SmInput[rowOffset + i]);
    }
    s_smMax[tid] = localMax;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = 128; s > 0; s >>= 1)
    {
        if (tid < s) s_smMax[tid] = max(s_smMax[tid], s_smMax[tid + s]);
        GroupMemoryBarrierWithGroupSync();
    }
    float maxVal = s_smMax[0];
    GroupMemoryBarrierWithGroupSync();

    // Pass 2: Sum Exp
    float localExpSum = 0.0f;
    for (uint j = tid; j < RowLength; j += 256)
    {
        localExpSum += exp(SmInput[rowOffset + j] - maxVal);
    }
    s_smSum[tid] = localExpSum;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint e = 128; e > 0; e >>= 1)
    {
        if (tid < e) s_smSum[tid] += s_smSum[tid + e];
        GroupMemoryBarrierWithGroupSync();
    }
    float invSum = 1.0f / max(s_smSum[0], 1e-12f);

    // Pass 3: Write Output
    for (uint k = tid; k < RowLength; k += 256)
    {
        SmOutput[rowOffset + k] = exp(SmInput[rowOffset + k] - maxVal) * invSum;
    }
}
";
    }
}
