#ifndef  RIGID_BODY_2_H
#define  RIGID_BODY_2_H

#include <thrust/device_vector.h>

#include "Renderer/Mesh/TriangleMesh.h"
#include "RigidBodyDeviceData2.cuh"
#include "Simulation/GPUDFSPH/RigidBody/DensityMap/DensityMap2.cuh"

namespace vfd
{
	struct RigidBody2Description
	{
		glm::mat4 Transform; // TODO: Use the transform component
		glm::uvec3 CollisionMapResolution = { 10, 10, 10 };
		std::string SourceMesh;

		bool Inverted;
		float Padding;
	};

	struct RigidBody2 : public RefCounted
	{
		RigidBody2(const RigidBody2Description& desc);

		RigidBody2DeviceData* GetDeviceData(unsigned int particleCount);

	private:
		RigidBody2Description m_Description;
		Ref<TriangleMesh> m_Mesh;
		DensityMap2 m_DensityMap;

		glm::mat3 m_Rotation;

		thrust::device_vector<glm::vec3> m_BoundaryXJ;
		thrust::device_vector<float> m_BoundaryVolume;
	};
}

#endif // !RIGID_BODY_2_H