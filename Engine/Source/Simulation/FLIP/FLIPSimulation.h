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
		int Resolution; 

		float TimeStep;
		float Viscosity;
		float CFLConditionNumber;
		float MinFrac;
		float RatioPICToFLIP;
	};

	struct FLIPParticle {
		glm::vec3 Position;
		glm::vec3 Velocity; 

		FLIPParticle() {}
		FLIPParticle(glm::vec3 p) : Position(p) {}
		FLIPParticle(glm::vec3 p, glm::vec3 v) : Position(p), Velocity(v) {}
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

		float CalculateCFL();

		void UpdateLiquidSDF();

		void AdvectVelocityField();
		void AdvectVelocityFieldU(Array3D<bool>& fluidCellGrid);
		void AdvectVelocityFieldV(Array3D<bool>& fluidCellGrid);
		void AdvectVelocityFieldW(Array3D<bool>& fluidCellGrid);
		void CalculateVelocityScalarField(Array3D<float>& field, Array3D<bool>& isValueSet, int dir);
		void ExtrapolateVelocityField(MACVelocityField& vfield, ValidVelocityComponentGrid& valid);

		void AddBodyForce(float dt); 

		void ApplyViscosity(float dt);

		void Project(float dt); 
		void ComputeWeights();
		void ApplyPressure(float dt, Array3D<float>& pressureGrid);
		Array3D<float> SolvePressure(float dt);
		void ConstrainVelocityField();
		void AdvectFluidParticles(float dt);
		void UpdateFluidParticleVelocities();
		glm::vec3 TraceRK2(glm::vec3 position, float dt); 
		glm::vec3 GetVelocity(glm::vec3 position); 
	public:
		bool paused = true;
	private:
		FLIPSimulationDescription m_Description;
		FLIPSimulationParameters m_Parameters; // device-side data

		Ref<VertexBuffer> m_PositionVBO;
		Ref<VertexArray> m_PositionVAO;

		bool m_Initialized = false;

		// MACVelocityField m_MACVelocity;
		MACVelocityField m_MACVelocity;
		MACVelocityField m_SavedVelocityField;

		WeightGrid m_WeightGrid;
		Array3D<float> m_Viscosity;

		ParticleLevelSet m_LiquidSDF;
		MeshLevelSet m_SolidSDF;

		ValidVelocityComponentGrid m_ValidVelocities;

		std::vector<FLIPParticle> m_Particles; // Starting positions, used for resetting the simulation
	};
}

#endif // !FLIP_SIMULATION_H