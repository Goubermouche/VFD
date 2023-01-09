#ifndef GPU_DFSPH_SIMULATOR_H
#define GPU_DFSPH_SIMULATOR_H

#include "DFSPHImplementation.h"
#include "Renderer/Renderer.h"
#include "GPUDFSPHSimulationDescription.h"
#include "Core/Cryptography/UUID.h"

namespace vfd
{
	// Wrapper around the GPU DFSPH simulator interface
	class GPUDFSPHSimulation : public RefCounted
	{
	public:
		GPUDFSPHSimulation(GPUDFSPHSimulationDescription& desc);
		~GPUDFSPHSimulation();

		void Simulate(std::vector<Ref<RigidBody>>& rigidBodies);
		void OnUpdate();
		const Ref<VertexArray>& GetVertexArray();

		// Getters
		unsigned int GetParticleCount();
		float GetParticleRadius() const;
		float GetMaxVelocityMagnitude() const;
		float GetCurrentTimeStepSize() const;
		const ParticleSearch* GetParticleSearch() const;
		const GPUDFSPHSimulationDescription& GetDescription() const;
		void SetDescription(const GPUDFSPHSimulationDescription& desc);
		const DFSPHSimulationInfo& GetInfo() const;
		PrecomputedDFSPHCubicKernel& GetKernel();
		const std::vector<Ref<RigidBody>>& GetRigidBodies() const;

		void Reset();
	public:
		bool paused = false;
	private:
		Ref<DFSPHImplementation> m_Implementation;

		bool m_Initialized = false;
	};
}

#endif