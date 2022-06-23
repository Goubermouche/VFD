#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, cuda
#include <cuda_runtime.h>
#include "FluidEngine/Compute/Utility/CUDAGLInterop.h"
#include "FluidEngine/Compute/Utility/HelperFunctions.h"
#include "FluidEngine/Compute/Utility/CUDAHelper.h"

extern "C" {
	 __global__ void simple_vbo_kernel(float4* pos, unsigned int width, unsigned int height, float time);
	 void launch_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, float time);
}