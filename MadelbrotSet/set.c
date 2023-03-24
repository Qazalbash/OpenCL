#ifndef SET_GAURD
#define SET_GAURD

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <stdio.h>

int main(int argc, char** argv) {
    const size_t N = 1024UL;

    size_t i, j;

    int* rz = (int*)malloc(N * N * sizeof(int));
    int* iz = (int*)malloc(N * N * sizeof(int));
    int* R  = (int*)malloc(N * N * sizeof(int));
    int* G  = (int*)malloc(N * N * sizeof(int));
    int* B  = (int*)malloc(N * N * sizeof(int));

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++) {
            rz[N * i + j] = (int)j;
            iz[N * i + j] = (int)i;
        }

    const char* kernel_file = "../kernel.cl";
    const char* kernel_name = "set";

    cl_uint work_dim = 1;

    size_t wd = 1;
    while (wd < N * N) wd <<= 1;
    wd = wd < 64 ? 64 : wd;

    printf("Work size: %lu\n", wd);
    const size_t* global_work_size = (const size_t[]){wd};
    const size_t* local_work_size  = (const size_t[]){64};

    cl_device_id device_id;

    cl_int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not get device.", err);
        return -1;
    }

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create context.", err);
        return -1;
    }

    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create command queue.", err);
        return -1;
    }

    FILE* fp = fopen(kernel_file, "r");
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);

    if (size == 0) {
        printf("Error: %d. kernel file has no function.", err);
        return -1;
    }

    fseek(fp, 0, SEEK_SET);
    char* source = (char*)malloc(size);
    fread(source, 1, size, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, &size, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create program.", err);
        return -1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not build program.", err);
        return -1;
    }

    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create kernel.", err);
        return -1;
    }

    cl_mem rz_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * N * N, NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    cl_mem iz_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * N * N, NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    cl_mem r_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * N, NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    cl_mem g_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * N, NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N * N, NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    cl_mem N_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    err = clEnqueueWriteBuffer(queue, rz_mem_obj, CL_TRUE, 0, sizeof(float) * N * N, rz, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, iz_mem_obj, CL_TRUE, 0, sizeof(float) * N * N, iz, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, N_mem_obj, CL_TRUE, 0, sizeof(size_t), &N, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not write buffer.", err);
        return -1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(rz_mem_obj), (void*)&rz_mem_obj);
    err |= clSetKernelArg(kernel, 1, sizeof(iz_mem_obj), (void*)&iz_mem_obj);
    err |= clSetKernelArg(kernel, 2, sizeof(r_mem_obj), (void*)&r_mem_obj);
    err |= clSetKernelArg(kernel, 3, sizeof(g_mem_obj), (void*)&g_mem_obj);
    err |= clSetKernelArg(kernel, 4, sizeof(b_mem_obj), (void*)&b_mem_obj);
    err |= clSetKernelArg(kernel, 5, sizeof(N_mem_obj), (void*)&N_mem_obj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not set kernel arguments.", err);
        return -1;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not enqueue kernel.", err);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, r_mem_obj, CL_TRUE, 0, sizeof(int) * N * N, R, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, g_mem_obj, CL_TRUE, 0, sizeof(int) * N * N, G, 0, NULL, NULL);
    err |= clEnqueueReadBuffer(queue, b_mem_obj, CL_TRUE, 0, sizeof(int) * N * N, B, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not read buffer.", err);
        return -1;
    }

    FILE* file;
    file = fopen("mandelbrot.ppm", "w");
    fprintf(file, "P3\n%lu %lu\n255\n", N, N);

    for (i = 0; i < N * N; i++) fprintf(file, "%d %d %d\n", R[i], G[i], B[i]);

    fclose(file);

    clFinish(queue);

    err = clReleaseKernel(kernel);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release kernel.", err);
        return -1;
    }

    err = clReleaseProgram(program);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release program.", err);
        return -1;
    }

    err = clReleaseCommandQueue(queue);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release command queue.", err);
        return -1;
    }

    err = clReleaseContext(context);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release context.", err);
        return -1;
    }

    err = clReleaseMemObject(rz_mem_obj);
    err |= clReleaseMemObject(iz_mem_obj);
    err |= clReleaseMemObject(r_mem_obj);
    err |= clReleaseMemObject(g_mem_obj);
    err |= clReleaseMemObject(b_mem_obj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release buffer.", err);
        return -1;
    }

    free(source);
    free(rz);
    free(iz);
    free(R);
    free(G);
    free(B);

    return 0;
}

#endif  // SET_GAURD