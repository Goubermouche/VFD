#ifndef  RIGID_BODY_H
#define  RIGID_BODY_H

#include <thrust/device_vector.h>

#include "Renderer/Mesh/TriangleMesh.h"
#include "Simulation/GPUDFSPH/RigidBody/RigidBodyDeviceData.cuh"
#include "Simulation/GPUDFSPH/DensityMap/DensityMap.cuh"
#include "Simulation/GPUDFSPH/DFSPHSimulationInfo.h"
#include "Simulation/GPUDFSPH/Kernel/DFSPHKernels.h"

namespace vfd
{
	struct RigidBodyDescription
	{
		bool Inverted;
		float Padding;

		glm::uvec3 CollisionMapResolution = { 10, 10, 10 };

		glm::mat4 Transform;
		Ref<TriangleMesh> Mesh;
	};

	struct RigidBody : public RefCounted
	{
		RigidBody(const RigidBodyDescription& desc, const DFSPHSimulationInfo& info, PrecomputedDFSPHCubicKernel& kernel);

		RigidBodyDeviceData* GetDeviceData();
		const RigidBodyDescription& GetDescription();
		const BoundingBox<glm::dvec3>& GetBounds() const;
	private:
		RigidBodyDescription m_Description;
		Ref<DensityMap> m_DensityMap;

		thrust::device_vector<glm::vec3> m_BoundaryXJ;
		thrust::device_vector<float> m_BoundaryVolume;
	};
}

#endif // !RIGID_BODY_H