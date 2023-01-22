#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>

int main(int argc, char** argv) {
    cl_int err;

    cl_device_id device_id;

    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

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

    FILE* fp = fopen("kernel.cl", "r");
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);

    if (size == 0) {
        printf("Error: %d. kernel file has no function.", 1);
        return -1;
    }

    fseek(fp, 0, SEEK_SET);
    char* source = (char*)malloc(size);
    fread(source, 1, size, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(
        context, 1, (const char**)&source, &size, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create program.", err);
        return -1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not build program.", err);
        return -1;
    }

    cl_kernel kernel = clCreateKernel(program, "vec_add", &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create kernel.", err);
        return -1;
    }

    const size_t N = 128;

    int* a = (int*)malloc(N * sizeof(int));
    int* b = (int*)malloc(N * sizeof(int));
    int* c = (int*)malloc(N * sizeof(int));

    for (size_t i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }

    cl_mem a_memobj =
        clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    cl_mem b_memobj =
        clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(int), NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    cl_mem c_memobj =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(int), NULL, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create buffer.", err);
        return -1;
    }

    err = clEnqueueWriteBuffer(queue, a_memobj, CL_TRUE, 0, N * sizeof(int), a,
                               0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not write buffer.", err);
        return -1;
    }

    err = clEnqueueWriteBuffer(queue, b_memobj, CL_TRUE, 0, N * sizeof(int), b,
                               0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not write buffer.", err);
        return -1;
    }

    err = clSetKernelArg(kernel, 0, sizeof(a_memobj), (void*)&a_memobj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not set kernel arguments.", err);
        return -1;
    }

    err = clSetKernelArg(kernel, 1, sizeof(b_memobj), (void*)&b_memobj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not set kernel arguments.", err);
        return -1;
    }

    err = clSetKernelArg(kernel, 2, sizeof(c_memobj), (void*)&c_memobj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not set kernel arguments.", err);
        return -1;
    }

    size_t global_item_size = N;
    size_t local_item_size  = 8;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size,
                                 &local_item_size, 0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not enqueue kernel.", err);
        return -1;
    }

    err = clEnqueueReadBuffer(queue, c_memobj, CL_TRUE, 0, N * sizeof(int), c,
                              0, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not read buffer.", err);
        return -1;
    }

    for (size_t i = 0; i < N; i++) printf("%d + %d = %d\n", a[i], b[i], c[i]);

    err = clFinish(queue);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not finish queue.", err);
        return -1;
    }

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

    err = clReleaseMemObject(a_memobj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release buffer.", err);
        return -1;
    }

    err = clReleaseMemObject(b_memobj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release buffer.", err);
        return -1;
    }

    err = clReleaseMemObject(c_memobj);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not release buffer.", err);
        return -1;
    }

    return 0;
}