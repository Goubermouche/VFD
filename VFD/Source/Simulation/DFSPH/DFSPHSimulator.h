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
		DFSPHSimulation(const DFSPHSimulationDescription& desc);
		~DFSPHSimulation();

		void Simulate();
		const Ref<VertexArray>& GetVertexArray();

		// Setters
		void SetFluidObjects(const std::vector<Ref<FluidObject>>& fluidObjects);
		void SetRigidBodies(const std::vector<Ref<RigidBody>>& rigidBodies);
		void SetFlowLineCount(unsigned int count);

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
		bool& GetRenderParticles();
		bool& GetRenderFlowLines();
		unsigned int GetFlowLineSampleCount() const;
		const std::vector<unsigned int>& GetFlowLineIndices() const;
	private:
		void RecomputeFlowLineIndices();
	private:
		Ref<DFSPHImplementation> m_Implementation;
		std::vector<unsigned int> m_FlowLineIndices;

		bool m_RenderParticles = true;
		bool m_RenderFlowLines = false;
		unsigned int m_FlowLineSampleCount = 100u;



		bool m_Initialized = false;
	};
}

#endif