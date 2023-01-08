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

		void OnUpdate();
		const Ref<VertexArray>& GetVertexArray();

		// Getters
		unsigned int GetParticleCount();
		float GetParticleRadius() const;
		Ref<Material>& GetRigidBodyMaterial();
		float GetMaxVelocityMagnitude() const;
		float GetCurrentTimeStepSize() const;
		std::vector<Ref<RigidBody>>& GetRigidBodies();
		const ParticleSearch* GetParticleSearch() const;
		const GPUDFSPHSimulationDescription& GetDescription() const;
		void SetDescription(const GPUDFSPHSimulationDescription& desc);

		void Reset();
	public:
		bool paused = false;
	private:
		Ref<DFSPHImplementation> m_Implementation;

		std::vector<Ref<RigidBody>> m_RigidBodies; // TEMP, this will be rendered as regular entities
		Ref<Material> m_RigidBodyMaterial; // TEMP

		bool m_Initialized = false;
	};
}

#endif