#ifndef RIGID_BODY_DEVICE_DATA_2_H
#define RIGID_BODY_DEVICE_DATA_2_H

#include "Simulation/GPUDFSPH/DensityMap/DensityMapDeviceData.cuh"

namespace vfd
{
	struct RigidBodyDeviceData
	{
		DensityMapDeviceData* Map;

		glm::vec3 Position;
		glm::mat3 Rotation;
		glm::vec3* BoundaryXJ;
		float* BoundaryVolume;
	};
}

#endif // !RIGID_BODY_DEVICE_DATA_2_H