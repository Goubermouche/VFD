#ifndef GPUSDF_CU
#define GPUSDF_CU

#include "pch.h"
#include "GPUSDF.cuh"
#include <cuda_gl_interop.h>

namespace vfd {
    //static __global__  void TestKernelArr(Arr<Particle> arr) {
    //    const uint32_t index = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

    //    arr[index].x = index;
    //    arr[index].y = index;
    //}

    void TestCUDA()
    {
        
    }
}

#endif // !GPUSDF_CU
