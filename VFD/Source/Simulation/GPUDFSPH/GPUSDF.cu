#ifndef GPUSDF_CU
#define GPUSDF_CU

#include "pch.h"
#include "GPUSDF.cuh"
#include <cuda_gl_interop.h>

namespace vfd {
    struct Test {
        int x, y;
        int* data;
    };

    static __global__  void TestKernel(Test* d) {
	    const uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

        printf("\nkernel: %d\n", index);
        printf("%d %d\n", ++d->x, ++d->y);
        printf("%d %d %d\n", ++d->data[0], ++d->data[1], ++d->data[2]);
    }

    static __global__  void TestKernelArr(Arr<int> arr) {
        const uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

        printf("\nDEVICE:\n");
        printf("size: %d\n", arr.GetSize());
        for (size_t i = 0; i < arr.GetSize(); i++)
        {
            printf("array: %d\n", arr.At(i));
        }

        arr.At(0) = 3;
        arr.At(1) = 30;
        arr.At(2) = 300;
        // arr.PushBack(999);
    }

    void TestCUDA()
    {
        //{
        //    int* hostData = new int[3]{ 1, 2, 3 };
        //    Test* host = new Test{ 10, 20, hostData };

        //    int* deviceData = nullptr;
        //    Test* device = nullptr;

        //    printf("\nhost:\n");
        //    printf("%d %d\n", host->x, host->y);
        //    printf("%d %d %d\n", host->data[0], host->data[1], host->data[2]);

        //    COMPUTE_SAFE(cudaMalloc(&device, sizeof(Test)));
        //    COMPUTE_SAFE(cudaMalloc(&deviceData, 3 * sizeof(int)));

        //    COMPUTE_SAFE(cudaMemcpy(device, host, sizeof(Test), cudaMemcpyHostToDevice));
        //    COMPUTE_SAFE(cudaMemcpy(deviceData, host->data, 3 * sizeof(int), cudaMemcpyHostToDevice));
        //    COMPUTE_SAFE(cudaMemcpy(&(device->data), &deviceData, sizeof(int*), cudaMemcpyHostToDevice));

        //    TestKernel << < 1, 1 >> > (device);
        //    COMPUTE_SAFE(cudaDeviceSynchronize());

        //    COMPUTE_SAFE(cudaMemcpy(host, device, sizeof(Test), cudaMemcpyDeviceToHost));
        //    host->data = hostData;
        //    COMPUTE_SAFE(cudaMemcpy(host->data, deviceData, 3 * sizeof(int), cudaMemcpyDeviceToHost));

        //    printf("\nhost:\n");
        //    printf("%d %d\n", host->x, host->y); // works
        //    printf("%d %d %d\n", host->data[0], host->data[1], host->data[2]);
        //}
     
        {
            Arr<int> arr(3);

            arr.PushBack(1);
            arr.PushBack(2);
            arr.PushBack(3);

            printf("\nHOST:\n");
            printf("size: %d\n", arr.GetSize());
            for (size_t i = 0; i < arr.GetSize(); i++)
            {
                printf("array: %d\n", arr.At(i));
            }

            COMPUTE_SAFE(cudaDeviceSynchronize());
            TestKernelArr<<<1, 1>>>(arr);
            COMPUTE_SAFE(cudaDeviceSynchronize());

            printf("\nHOST:\n");
            printf("size: %d\n", arr.GetSize());
            for (size_t i = 0; i < arr.GetSize(); i++)
            {
                printf("array: %d\n", arr.At(i));
            }
        }
    }
}

#endif // !GPUSDF_CU
