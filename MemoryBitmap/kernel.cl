__kernel void MatrixMulKernel(__global const float* d_M, __global const float* d_N, __global float* d_P,
                              __global const int* Width) {
    const int Row = get_group_id(1) * get_local_size(1) + get_local_id(1);
    const int Col = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if ((Row < *Width) && (Col < *Width)) {
        float Pvalue = 0.0f;
        for (int k = 0; k < *Width; ++k) Pvalue += d_M[Row * *Width + k] * d_N[k * *Width + Col];
        d_P[Row * *Width + Col] = Pvalue;
    }
}

__kernel void MatrixMulKernelTiled(__global float* d_M, __global float* d_N, __global float* d_P,
                                   __global const int* Width) {
    const int TILE_WIDTH = 16;

    __local float Mds[TILE_WIDTH][TILE_WIDTH];
    __local float Nds[TILE_WIDTH][TILE_WIDTH];

    const int bx = get_group_id(0);
    const int by = get_group_id(1);
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);

    const int Row    = by * TILE_WIDTH + ty;
    const int Col    = bx * TILE_WIDTH + tx;
    float     Pvalue = 0;

    for (int m = 0; m < (TILE_WIDTH + *Width - 1) / TILE_WIDTH; ++m) {
        if (m * TILE_WIDTH + tx < *Width && Row < *Width)
            Mds[ty][tx] = d_M[Row * *Width + m * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if (m * TILE_WIDTH + ty < *Width && Col < *Width)
            Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * *Width + Col];
        else
            Nds[ty][tx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (Row < *Width && Col < *Width) d_P[Row * *Width + Col] = Pvalue;
}