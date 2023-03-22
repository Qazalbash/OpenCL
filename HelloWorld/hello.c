#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>
#include <stdio.h>

int main(int argc, char** argv) {
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

    FILE* fp = fopen("kernel.cl", "r");
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

    cl_kernel kernel = clCreateKernel(program, "greeting", &err);

    if (err != CL_SUCCESS) {
        printf("Error: %d. OpenCL could not create kernel.", err);
        return -1;
    }

    cl_uint       work_dim         = 2;
    const size_t* global_work_size = (const size_t[]){8, 8};
    const size_t* local_work_size  = (const size_t[]){4};

    clEnqueueNDRangeKernel(queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, NULL);

    clFinish(queue);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(source);

    return 0;
}