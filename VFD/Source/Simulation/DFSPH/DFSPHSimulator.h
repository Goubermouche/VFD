#ifndef DFSPH_SIMULATOR_H
#define DFSPH_SIMULATOR_H

#include "Simulation/DFSPH/DFSPHImplementation.h"
#include "Simulation/DFSPH/Structures/DFSPHSimulationDescription.h"

namespace vfd
{
	// Wrapper around the GPU DFSPH simulator interface
	class DFSPHSimulation : public RefCounted
	{
	public:
		DFSPHSimulation(DFSPHSimulationDescription& desc);
		~DFSPHSimulation();

		void Simulate();
		const Ref<VertexArray>& GetVertexArray();

		// Setters
		void SetFluidObjects(const std::vector<Ref<FluidObject>>& fluidObjects);
		void SetRigidBodies(const std::vector<Ref<RigidBody>>& rigidBodies);

		// Getters
		DFSPHImplementation::SimulationState GetSimulationState() const;
		unsigned int GetParticleCount();
		float GetParticleRadius() const;
		float GetMaxVelocityMagnitude() const;
		float GetCurrentTimeStepSize() const;
		const ParticleSearch& GetParticleSearch() const;
		const DFSPHSimulationDescription& GetDescription() const;
		void SetDescription(const DFSPHSimulationDescription& desc);
		const DFSPHSimulationInfo& GetInfo() const;
		PrecomputedDFSPHCubicKernel& GetKernel();
		const std::vector<Ref<RigidBody>>& GetRigidBodies() const;
		unsigned int GetRigidBodyCount() const;
		const DFSPHDebugInfo& GetDebugInfo() const;
		Ref<DFSPHParticleBuffer> GetParticleFrameBuffer();
	private:
		Ref<DFSPHImplementation> m_Implementation;

		bool m_Initialized = false;
	};
}

#endif