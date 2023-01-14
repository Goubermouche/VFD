#ifndef RIGID_BODY_DEVICE_DATA_2_H
#define RIGID_BODY_DEVICE_DATA_2_H

#include "Utility/SDF/SDFDeviceData.cuh"

namespace vfd
{
	struct RigidBodyDeviceData
	{
		SDFDeviceData* Map;

		glm::vec3* BoundaryXJ;
		float* BoundaryVolume;
	};
}

#endif // !RIGID_BODY_DEVICE_DATA_2_H