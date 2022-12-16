#ifndef GPU_DFSPH_SIMULATOR_H
#define GPU_DFSPH_SIMULATOR_H

#include "DFSPHImplementation.h"
#include "Renderer/Renderer.h"
#include "RigidBody/RigidBody.h"
#include "GPUDFSPHSimulationDescription.h"

namespace vfd
{
	class DFSPHImplementation;

	// Wrapper around the GPU DFSPH simulator interface
	class GPUDFSPHSimulation : public RefCounted
	{
	public:
		GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc);
		~GPUDFSPHSimulation();

		void OnUpdate();
		const Ref<VertexArray>& GetVertexArray();

		// Getters
		unsigned int GetParticleCount();
		float GetParticleRadius() const;
		std::vector<Ref<RigidBody>>& GetRigidBodies();
		Ref<Material>& GetRigidBodyMaterial();
		float GetMaxVelocityMagnitude() const;
		float GetCurrentTimeStepSize() const;

		void Reset();
	public:
		bool paused = false;
	private:
		GPUDFSPHSimulationDescription m_Description;
		Ref<DFSPHImplementation> m_Implementation;

		std::vector<Ref<RigidBody>> m_RigidBodies;
		Ref<Material> m_RigidBodyMaterial; // TEMP

		bool m_Initialized = false;
	};
}

#endif