#ifndef  RIGID_BODY_H
#define  RIGID_BODY_H

#include <thrust/device_vector.h>

#include "Renderer/Mesh/TriangleMesh.h"
#include "Simulation/GPUDFSPH/RigidBody/RigidBodyDeviceData.cuh"
#include "Simulation/GPUDFSPH/DensityMap/DensityMap.cuh"

namespace vfd
{
	struct RigidBodyDescription
	{
		glm::mat4 Transform; // TODO: Use the transform component
		glm::uvec3 CollisionMapResolution = { 10, 10, 10 };
		std::string SourceMesh;

		bool Inverted;
		float Padding;
	};

	struct RigidBody : public RefCounted
	{
		RigidBody(const RigidBodyDescription& desc);

		RigidBodyDeviceData* GetDeviceData(unsigned int particleCount);

		const Ref<TriangleMesh>& GetMesh();
		const glm::mat4& GetTransform();
	private:
		RigidBodyDescription m_Description;
		Ref<TriangleMesh> m_Mesh;
		DensityMap m_DensityMap;

		glm::mat3 m_Rotation;

		thrust::device_vector<glm::vec3> m_BoundaryXJ;
		thrust::device_vector<float> m_BoundaryVolume;
	};
}

#endif // !RIGID_BODY_H