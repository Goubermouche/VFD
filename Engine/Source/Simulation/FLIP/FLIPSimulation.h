#ifndef FLIP_SIMULATION_H
#define FLIP_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "Simulation/FLIP/FLIPSimulationParameters.cuh"
#include "Renderer/Mesh/TriangleMesh.h"

namespace fe {
	// TODO: 
	// - proper struct deallocation

	// NOTE: 
	// - this simulation method is a Cuda implementation of this repository: https://github.com/rlguy/FLIPViscosity3D.
	// - the code is currently very cursed, this will be fixed in upcoming commits. 

	struct FLIPSimulationDescription {
		int MeshLevelSetExactBand;
		INT Resolution; 

		float TimeStep;

		uint32_t SubStepCount;

	};

	class FLIPSimulation : public RefCounted
	{
	public:
		FLIPSimulation(const FLIPSimulationDescription& desc);
		~FLIPSimulation();

		void OnUpdate();
		void OnRenderTemp();

		void AddBoundary(const std::string& filepath, bool inverted = false);
		void AddLiquid(const std::string& filepath);

		const Ref<VertexArray>& GetVAO() {
			return m_PositionVAO;
		}

		FLIPSimulationParameters GetParameters() const {
			return m_Parameters;
		}
	private:
		void InitMemory();
		void FreeMemory();

	 	TriangleMesh GetBoundaryTriangleMesh();
		void InitBoundary();
	public:
		bool paused = false;
	private:
		FLIPSimulationDescription m_Description;
		FLIPSimulationParameters m_Parameters; // device-side data

		Ref<VertexBuffer> m_PositionVBO;
		Ref<VertexArray> m_PositionVAO;

		bool m_Initialized = false;

		// MACVelocityField m_MACVelocity;
		MACVelocityField m_MACVelocity;
		MACVelocityField m_MACVelocityDevice;

		WeightGrid m_WeightGrid;
		Array3D<float> m_Viscosity;

		ParticleLevelSet m_LiquidSDF;
		MeshLevelSet m_SolidSDF;

		ValidVelocityComponent m_ValidVelocities;

		std::vector<glm::vec3> m_PositionCache; // Starting positions, used for resetting the simulation
	};
}

#endif // !FLIP_SIMULATION_H