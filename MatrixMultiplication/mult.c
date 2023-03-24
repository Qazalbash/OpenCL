#ifndef MAT_MUL_GAURD
#define MAT_MUL_GAURD

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 300
#define MAX_SOURCE_SIZE          (0x100000)

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    cl_int err;

    int k = 5;

    const size_t LIST_SIZE = k * k;

    float *A = (float *)malloc(LIST_SIZE * sizeof(float));
    float *B = (float *)malloc(LIST_SIZE * sizeof(float));
    float *C = (float *)malloc(LIST_SIZE * sizeof(float));

    size_t i;
    for (i = 0; i < LIST_SIZE; i++) {
        A[i] = rand() / (float)RAND_MAX;
        B[i] = rand() / (float)RAND_MAX;
    }

    for (i = 0; i < (size_t)k; i++) B[i * k] = 0.5f;

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

    FILE *fp = fopen("../kernel.cl", "r");
    fseek(fp, 0, SEEK_END);
    size_t size = ftell(fp);

    if (size == 0) {
        printf("Error: %d. kernel file has no function.", 1);
        return -1;
    }

    fseek(fp, 0, SEEK_SET);
    char *source = (char *)malloc(size);
    fread(source, 1, size, fp);
    fclose(fp);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &size, &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create program.", err);
        return -1;
    }

    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not build program.", err);
        return -1;
    }

    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(float), NULL, &err);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(float), NULL, &err);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(float), NULL, &err);
    cl_mem k_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &err);

    // Copy the lists A and B to their respective memory buffers
    err = clEnqueueWriteBuffer(queue, a_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(float), A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(float), B, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, k_mem_obj, CL_TRUE, 0, sizeof(int), &k, 0, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matrix_multiplication", &err);

    // Set the arguments of the kernel
    err = clSetKernelArg(kernel, 0, sizeof(a_mem_obj), (void *)&a_mem_obj);
    err |= clSetKernelArg(kernel, 1, sizeof(b_mem_obj), (void *)&b_mem_obj);
    err |= clSetKernelArg(kernel, 2, sizeof(c_mem_obj), (void *)&c_mem_obj);
    err |= clSetKernelArg(kernel, 3, sizeof(k_mem_obj), (void *)&k_mem_obj);

    // Execute the OpenCL kernel on the list
    const size_t global_item_size = LIST_SIZE;  // Process the entire lists

    const size_t local_item_size = k;  // Divide work items into groups of 64

    // Execute the kernel on the device
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, (const size_t *)&global_item_size,
                                 (const size_t *)&local_item_size,
                                 //  NULL, NULL,
                                 0, NULL, NULL);

    // Read the memory buffer C on the device to the local variable C
    err = clEnqueueReadBuffer(queue, c_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(float), C, 0, NULL, NULL);

    // Display the result to the screen

    for (i = 0; i < LIST_SIZE; i++) {
        if (i % k == 0) printf("\n");
        printf("%f ", A[i]);
    }
    printf("\n");

    for (i = 0; i < LIST_SIZE; i++) {
        if (i % k == 0) printf("\n");
        printf("%f ", B[i]);
    }
    printf("\n");

    for (i = 0; i < LIST_SIZE; i++) {
        if (i % k == 0) printf("\n");
        printf("%f ", C[i]);
    }
    printf("\n");

    // Clean up
    err = clFlush(queue);
    err |= clFinish(queue);
    err |= clReleaseKernel(kernel);
    err |= clReleaseProgram(program);
    err |= clReleaseMemObject(a_mem_obj);
    err |= clReleaseMemObject(b_mem_obj);
    err |= clReleaseMemObject(c_mem_obj);
    err |= clReleaseCommandQueue(queue);
    err |= clReleaseContext(context);

    free(source);
    free(A);
    free(B);
    free(C);

    return 0;
}

#endif