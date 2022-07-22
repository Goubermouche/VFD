#ifndef SPH_SIMULATION_H_
#define SPH_SIMULATION_H_

#include "FluidEngine/Renderer/Renderer.h"
#include "FluidEngine/Compute/GPUCompute.h"

#include "SimulationParameters.cuh"

#include "FluidEngine/Utility/Sampler/ParticleSampler.h"

namespace fe {
	struct ParticleVolumeDescription {
		std::string sourceMesh;

		glm::vec3 scale;
		glm::vec3 position;

		glm::ivec3 resolution;

		SampleMode sampleMode;
	};

	struct SPHSimulationDescription {
		float particleRadius;
		float homogenity;
		float restDensity;
		float stiffness;
		float viscosity;
		float timeStep;
		float globalDamping;
		float boundsStiffness;
		float boundsDamping;
		float boundsDampingCritical;

		unsigned int maxParticlesInCellCount;

		glm::vec3 gravity;
		glm::vec3 worldMin;
		glm::vec3 worldMax;

		std::vector<ParticleVolumeDescription> particleVolumes;
	};

	class SPHSimulation : public RefCounted
	{
	public:
		SPHSimulation(const SPHSimulationDescription& description);
		~SPHSimulation();

		void OnUpdate();
		void OnRender();

		SPHSimulationDescription GetDescription() const {
			return m_Description;
		}

		SimulationData GetData() const {
			return m_Data;
		}
	private:
		/// <summary>
		/// Sets the initial values for position, velocity, has, and cell start arrays and allocates the neccessary memory.
		/// </summary>
		void InitMemory();
		void FreeMemory();

		/// <summary>;
		/// Updates constant particle values based on the current params.
		/// </summary>
		void UpdateParticles();

		/// <summary>
		/// Updates constant grid values based on the current params.
		/// </summary>
		void UpdateGrid();

		std::vector<glm::vec4> LoadParticleVolumes();
		void SetArray(bool pos, const glm::vec4* data, int start, int count);
	public:
		bool m_Paused = false;
	private:
		glm::vec4* m_Position;
		glm::vec4* m_Velocity;
		glm::vec4* m_DeltaPosition[2];
		glm::vec4* m_DeltaVelocity[2];
		glm::vec4* m_SortedPosition;
		glm::vec4* m_SortedVelocity;

		uint32_t* m_ParticleHash;
		uint32_t* m_DeltaParticleHash[2];
		uint32_t* m_CellStart;
		uint32_t* m_DeltaCellStart;

		float* m_Pressure;
		float* m_Density;

		Ref<VertexBuffer> m_PositionVBO[2];
		Ref<VertexArray> m_PositionVAO[2];

		bool m_CurrentPositionRead;
		bool m_CurrentVelocityRead;
		bool m_CurrentPositionWrite;
		bool m_CurrentVeloctiyWrite;

		bool m_Initialized = false;

		Ref<Material> m_PointMaterial;

		SPHSimulationDescription m_Description;
		SimulationData m_Data;
	};
}

#endif // !SPH_SIMULATION_H_