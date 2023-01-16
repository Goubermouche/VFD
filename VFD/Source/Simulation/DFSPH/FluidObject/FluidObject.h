#ifndef FLUID_OBJECT_H
#define FLUID_OBJECT_H

#include "Renderer/Mesh/TriangleMesh.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Simulation/DFSPH/Structures/DFSPHSimulationInfo.h"

namespace vfd
{
	struct FluidObjectDescription
	{
		bool Inverted;
		glm::uvec3 Resolution = { 10u, 10u, 10u };
		SampleMode SampleMode;
		glm::vec3 Velocity = { 0.0f, 0.0f, 0.0f };

		glm::mat4 Transform;
		Ref<TriangleMesh> Mesh;
	};

	struct FluidObject : public RefCounted
	{
		FluidObject(const FluidObjectDescription& desc, const DFSPHSimulationInfo& info);

		const std::vector<glm::vec3>& GetPositions() const;
		unsigned int GetPositionCount() const;
		const glm::vec3& GetVelocity() const;
	private:
		std::vector<glm::vec3> m_Positions;
		FluidObjectDescription m_Description;
	};
}

#endif // !FLUID_OBJECT_H