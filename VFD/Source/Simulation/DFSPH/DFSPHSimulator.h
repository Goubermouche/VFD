#ifndef DFSPH_SIMULATOR_H
#define DFSPH_SIMULATOR_H

#include "DFSPHImplementation.h"
#include "Renderer/Renderer.h"
#include "DFSPHSimulationDescription.h"
#include "Core/Cryptography/UUID.h"

namespace vfd
{
	// Wrapper around the GPU DFSPH simulator interface
	class DFSPHSimulation : public RefCounted
	{
	public:
		DFSPHSimulation(DFSPHSimulationDescription& desc);
		~DFSPHSimulation();

		void Simulate(const std::vector<Ref<RigidBody>>& rigidBodies);
		void OnUpdate();
		const Ref<VertexArray>& GetVertexArray();

		// Getters
		unsigned int GetParticleCount();
		float GetParticleRadius() const;
		float GetMaxVelocityMagnitude() const;
		float GetCurrentTimeStepSize() const;
		const ParticleSearch* GetParticleSearch() const;
		const DFSPHSimulationDescription& GetDescription() const;
		void SetDescription(const DFSPHSimulationDescription& desc);
		const DFSPHSimulationInfo& GetInfo() const;
		PrecomputedDFSPHCubicKernel& GetKernel();
		const std::vector<Ref<RigidBody>>& GetRigidBodies() const;
		const DFSPHDebugInfo& GetDebugInfo() const;

		void Reset();
	public:
		bool paused = false;
	private:
		Ref<DFSPHImplementation> m_Implementation;

		bool m_Initialized = false;
	};
}

#endif