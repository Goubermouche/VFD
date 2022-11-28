#ifndef GPUSDF_CU
#define GPUSDF_CU

#include "pch.h"
#include "GPUSDF.cuh"
#include <cuda_gl_interop.h>

namespace vfd {
    struct Particle {
        float x;
        uint64_t y;
    };

    static __global__  void TestKernelArr(Arr<Particle> arr) {
        const uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

        arr[index].x = index;
        arr[index].y = index;
    }

    void TestCUDA()
    {
        //Arr<Particle> arr;

        //arr.AddElement(Particle{ 1, 10 });
        //arr.AddElement(Particle{ 2, 20 });
        //arr.AddElement(Particle{ 3, 30 });

        //printf("\nHOST:\n");
        //printf("size: %d\n", arr.GetSize());
        //for (size_t i = 0; i < arr.GetSize(); i++)
        //{
        //    printf("array: %.2f %d\n", arr[i].x, arr[i].y);
        //}

        //TestKernelArr<<<1, arr.GetSize() >> >(arr);
        //COMPUTE_SAFE(cudaDeviceSynchronize());

        //printf("\nHOST:\n");
        //printf("size: %d\n", arr.GetSize());
        //for (size_t i = 0; i < arr.GetSize(); i++)
        //{
        //    printf("array: %.2f %d\n", arr[i].x, arr[i].y);
        //}

        //arr.Free();
    }
}

#endif // !GPUSDF_CU
