#ifndef RIGID_BODY_IMPLEMENTATION_H
#define RIGID_BODY_IMPLEMENTATION_H

#include "DensityMap.cuh"

namespace vfd
{
	struct RigidBodyDescription;

	struct RigidBodyDeviceData
	{
	public:
		__host__ RigidBodyDeviceData() = default;
		__host__ RigidBodyDeviceData(const RigidBodyDescription& desc);
		__host__ ~RigidBodyDeviceData();

		__host__ __device__ __forceinline__ glm::vec3& GetBoundaryXJ(unsigned int i) const
		{
			return BoundaryXJ[i];
		}

		__host__ __device__ __forceinline__ float& GetBoundaryVolume(unsigned int i) const
		{
			return BoundaryVolume[i];
		}

		glm::mat3 Rotation;
		DensityMap* Map;

		glm::vec3* BoundaryXJ;
		float* BoundaryVolume;
	};
}

#endif // !RIGID_BODY_IMPLEMENTATION_H