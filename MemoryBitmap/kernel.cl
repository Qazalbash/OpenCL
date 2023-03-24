__kernel void WaveMakeImageKernel(__global unsigned char* ptr) {
    int x      = get_local_id(0) + get_group_id(0) * get_local_size(0);
    int y      = get_local_id(1) + get_group_id(1) * get_local_size(1);
    int offset = x + y * get_local_size(0) * get_num_groups(0);

    const float period = 128.0f;
    const float PI     = 3.14159f;

    float dx    = x - 256;
    float dy    = y - 256;
    float value = 255.0f * sin(sqrt(dx * dx + dy * dy) * 2.0f * PI / period);

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = value;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}