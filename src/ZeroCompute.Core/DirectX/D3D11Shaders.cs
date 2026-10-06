namespace ZeroCompute.Core.DirectX
{
    /// <summary>
    /// Embedded DirectCompute HLSL compute shaders (Shader Model 5.0) for accelerated tensor operations.
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
    if (id.x >= Count) return;
    float x = ActInput[id.x];
    float res = 0.0f;

    if (ActType == 0)
    {
        res = max(0.0f, x);
    }
    else if (ActType == 1)
    {
        res = x >= 0.0f ? x : 0.01f * x;
    }
    else if (ActType == 2)
    {
        const float sqrt2OverPi = 0.79788456f;
        const float coeff = 0.044715f;
        float inner = sqrt2OverPi * (x + coeff * x * x * x);
        res = 0.5f * x * (1.0f + tanh(inner));
    }
    else if (ActType == 3)
    {
        res = 1.0f / (1.0f + exp(-x));
    }
    else if (ActType == 4)
    {
        res = tanh(x);
    }

    ActOutput[id.x] = res;
}
";
    }
}
