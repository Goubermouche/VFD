#ifndef GPUSDF_CU
#define GPUSDF_CU

#include "pch.h"
#include "GPUSDF.cuh"
#include <cuda_gl_interop.h>

namespace vfd {
    static __global__  void TestKernelArr(Arr<int> arr) {
        const uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

        arr[0] = 999;

        printf("\nDEVICE:\n");
        printf("size: %d\n", arr.GetSize());
        for (size_t i = 0; i < arr.GetSize(); i++)
        {
            printf("array: %d\n", arr[i]);
        }

      
    }

    void TestCUDA()
    {
        Arr<int> arr(3);

        arr.PushBack(1);
        arr.PushBack(2);
        arr.PushBack(3);

        printf("\nHOST:\n");
        printf("size: %d\n", arr.GetSize());
        for (size_t i = 0; i < arr.GetSize(); i++)
        {
            printf("array: %d\n", arr[i]);
        }

        COMPUTE_SAFE(cudaDeviceSynchronize());
        TestKernelArr<<<1, 1>>>(arr);
        COMPUTE_SAFE(cudaDeviceSynchronize());

        printf("\nHOST:\n");
        printf("size: %d\n", arr.GetSize());
        for (size_t i = 0; i < arr.GetSize(); i++)
        {
            printf("array: %d\n", arr[i]);
        }

        arr.Free();
    }
}

#endif // !GPUSDF_CU
