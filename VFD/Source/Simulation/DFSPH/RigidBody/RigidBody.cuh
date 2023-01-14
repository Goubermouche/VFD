#ifndef  RIGID_BODY_H
#define  RIGID_BODY_H

#include "Renderer/Mesh/TriangleMesh.h"
#include "Utility/SDF/SDF.cuh"

#include "Simulation/DFSPH/RigidBody/RigidBodyDeviceData.cuh"
#include "Simulation/DFSPH/Structures/DFSPHSimulationInfo.h"
#include "Simulation/DFSPH/Kernel/DFSPHKernels.h"

#include <thrust/device_vector.h>

namespace vfd
{
	struct RigidBodyDescription
	{
		bool Inverted;
		float Padding;

		glm::uvec3 CollisionMapResolution = { 10u, 10u, 10u };

		glm::mat4 Transform;
		Ref<TriangleMesh> Mesh;
	};

	struct RigidBody : public RefCounted
	{
		RigidBody(const RigidBodyDescription& desc, const DFSPHSimulationInfo& info, PrecomputedDFSPHCubicKernel& kernel);
		~RigidBody();

		RigidBodyDeviceData* GetDeviceData();
		const RigidBodyDescription& GetDescription();
		const BoundingBox<glm::vec3>& GetBounds() const;
	private:
		RigidBodyDeviceData* d_DeviceData = nullptr;
		RigidBodyDescription m_Description;
		Ref<SDF> m_DensityMap;

		thrust::device_vector<glm::vec3> m_BoundaryXJ;
		thrust::device_vector<float> m_BoundaryVolume;
	};
}

#endif // !RIGID_BODY_H