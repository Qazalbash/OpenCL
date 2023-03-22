__kernel void matrix_multiplication(__global const float *A, __global const float *B, __global float *C,
                                    __global const int *N) {
    float sum = 0.0f;

    int tx = get_global_id(0), i = tx % *N, j = tx / *N;

    for (int k = 0; k < *N; k++) sum += A[i * *N + k] * B[k * *N + j];

    C[i * *N + j] = sum;
}