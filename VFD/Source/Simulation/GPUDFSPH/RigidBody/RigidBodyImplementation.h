#ifndef RIGID_BODY_IMPLEMENTATION_H
#define RIGID_BODY_IMPLEMENTATION_H

#include "DensityMap.cuh"

namespace vfd
{
	struct RigidBodyDescription;

	struct RigidBodyImplementation
	{
	public:
		__host__ RigidBodyImplementation() = default;
		__host__ RigidBodyImplementation(const RigidBodyDescription& desc);
		__host__ ~RigidBodyImplementation();

		__host__ __device__ __forceinline__ glm::vec3& GetBoundaryXJ(unsigned int i)
		{
			return BoundaryXJ[i];
		}

		__host__ __device__ __forceinline__ float& GetBoundaryVolume(unsigned int i)
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