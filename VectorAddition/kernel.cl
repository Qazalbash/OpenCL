__kernel void vec_add(__global const int *a, __global const int *b,
                      __global int *c) {
    int i = get_global_id(0);
    c[i]  = a[i] + b[i];
}