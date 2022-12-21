#ifndef POINT_SET_DEVICE_DATA_CUH
#define POINT_SET_DEVICE_DATA_CUH

namespace vfd {

	struct PointSetDeviceData {

		unsigned int* Counts;
		unsigned int* Offsets;
		unsigned int* Neighbors;

		__host__ __device__ unsigned int GetNeighborCount(unsigned int i) {
			return Counts[i];
		}

		__host__ __device__ unsigned int GetNeighbor(unsigned int i, unsigned int k) {
			return Neighbors[Offsets[i] + k];
		}

		__host__ __device__ unsigned int* GetNeighborList(const unsigned int i) {
			return &Neighbors[Offsets[i]];
		}
	};
}

#endif POINT_SET_DEVICE_DATA_CUH