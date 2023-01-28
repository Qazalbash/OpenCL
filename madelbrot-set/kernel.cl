__kernel void set(__global const int* rz, __global const int* iz,
                  __global int* Red, __global int* Green, __global int* Blue,
                  __global const size_t* N) {
    int i = get_global_id(0);

    // map [0, 512] to [-2, 2]
    float       alpha  = (float)rz[i] / (float)*N;
    const float c_real = alpha * 4.0f - 2.0f;
    alpha              = (float)iz[i] / (float)*N;
    const float c_imag = alpha * 4.0f - 2.0f;

    float real = c_real, imag = c_imag, real_ = 0.0f, imag_ = 0.0f, r_sq = 0.0f;

    int count = 0;

    // iterate until |z| > 2 or 100 iterations
    do {
        real_ = real * real - imag * imag + c_real;
        imag_ = 2.0f * real * imag + c_imag;
        r_sq  = real_ * real_ + imag_ * imag_;
        real  = real_;
        imag  = imag_;
    } while (r_sq <= 4.0f && count++ < 100);

    float P = 0.0f, Q = 25.0f, R = 100.0f, X = (float)count;

    Red[i]   = 225.0f * (X - Q) * (X - R) / ((P - Q) * (P - R));
    Green[i] = 225.0f * (X - P) * (X - R) / ((Q - P) * (Q - R));
    Blue[i]  = 225.0f * (X - P) * (X - Q) / ((R - P) * (R - Q));
}