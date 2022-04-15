#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    int deviceCount;
    cudaDeviceProp deviceProp;
    //Сколько устройств CUDA установлено на PC.
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n\n", deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        //Получаем информацию об устройстве
        cudaGetDeviceProperties(&deviceProp, i);
        //Выводим иформацию об устройстве
        printf("Device name: %s\n", deviceProp.name);
        printf("Total global memory: %d\n",
            deviceProp.totalGlobalMem);
        printf("Shared memory per block: %d\n",
            deviceProp.sharedMemPerBlock);
        printf("Registers per block: %d\n",
            deviceProp.regsPerBlock);
        printf("Warp size: %d\n", deviceProp.warpSize);
        printf("Memory pitch: %d\n", deviceProp.memPitch);
        printf("Max threads per block: %d\n",
            deviceProp.maxThreadsPerBlock);
        printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("Max grid size: x = %d, y = %d, z = %d\n",
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);
        printf("Clock rate: %d\n", deviceProp.clockRate);
        printf("Total constant memory: %d\n",
            deviceProp.totalConstMem);
        printf("Compute capability: %d.%d\n", deviceProp.major,
            deviceProp.minor);
        printf("Texture alignment: %d\n",
            deviceProp.textureAlignment);
        printf("Device overlap: %d\n", deviceProp.deviceOverlap);
        printf("Multiprocessor count: %d\n",
            deviceProp.multiProcessorCount);
        printf("Kernel execution timeout enabled: %s\n",
            deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
    }

    cout << endl;

    const int arraySize = 10;
    const int a[arraySize] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    const int b[arraySize] = { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cout << "a + b = " << "{ ";
    for (int i = 0; i < arraySize; ++i)
    {
        cout << c[i];
        if (i < arraySize - 1)
        {
            cout << ", ";
        }
    }
    cout << " }";
    cout << endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
