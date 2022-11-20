#ifndef GPU_SDF_CUH
#define GPU_SDF_CUH

namespace vfd {
	extern "C" {
		void ComputeSDF(float* V, int V_size,
			int* F, int F_size,
			float* sdf, int D1, int D2, int D3, float grid_size, float* min_corner);
	}
}

#endif // !GPU_SDF_CUH