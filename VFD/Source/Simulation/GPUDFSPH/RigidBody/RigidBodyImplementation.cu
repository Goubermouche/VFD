#include "pch.h"
#include "RigidBodyImplementation.h"

#include "RigidBody.h"

namespace vfd
{
	__host__ RigidBodyImplementation::RigidBodyImplementation(const RigidBodyDescription& desc)
	{
		glm::vec3 scale;
		for (int i = 0; i < 3; i++)
		{
			scale[i] = glm::length(glm::vec3(desc.Transform[i]));
		}

		Rotation = glm::mat3(
			glm::vec3(desc.Transform[0]) / scale[0],
			glm::vec3(desc.Transform[1]) / scale[1],
			glm::vec3(desc.Transform[2]) / scale[2]
		);

		Map = new DensityMap("Resources/b.cdm");
	}

	__host__ RigidBodyImplementation::~RigidBodyImplementation()
	{
		ERR("delete")
	}
}