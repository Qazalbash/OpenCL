/*
    get_group_id(uint dimindx)    =   blockIdx.[xyz]
    get_local_size(uint dimindx)  =   blockDim.[xyz]
    get_local_id(uint dimindx)    =   threadIdx.[xyz]
    get_num_groups(uint dimindx)  =   gridDim.[xyz]
*/

__kernel void PictureKernel(__global const unsigned char *d_Pin,
                            __global unsigned char       *d_Pout,
                            __global const int *n, __global const int *m,
                            __global const float *brightness) {
    // Calculate the row #
    // int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Row = get_group_id(1) * get_local_size(1) + get_local_id(1);

    // Calculate the column #
    // int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Col = get_group_id(0) * get_local_size(0) + get_local_id(0);

    if ((Row < *m) && (Col < *n)) {
        int offset = (Row * *n) + Col;
        // this is to flip the output image
        int offset2 = (((*n - 1) - Row) * *n) + Col;

        d_Pout[offset2] = d_Pin[offset] * *brightness;
    }
}
