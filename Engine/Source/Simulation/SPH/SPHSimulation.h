#ifndef SPH_SIMULATION_H
#define SPH_SIMULATION_H

#include "Renderer/Renderer.h"
#include "Compute/GPUCompute.h"
#include "SimulationParameters.cuh"
#include "Utility/Sampler/ParticleSampler.h"

namespace fe {
	struct ParticleVolumeDescription {
		std::string SourceMesh;
		glm::vec3 Scale;
		glm::vec3 Position;
		glm::uvec3 Resolution;
		SampleMode SampleMode;
	};

	struct SPHSimulationDescription {
		float ParticleRadius;
		float Homogeneity;
		float RestDensity;
		float Stiffness;
		float Viscosity;
		float TimeStep;
		float GlobalDamping;
		float BoundsStiffness;
		float BoundsDamping;
		float BoundsDampingCritical;

		unsigned int MaxParticlesInCellCount;

		glm::vec3 Gravity;
		glm::vec3 WorldMin;
		glm::vec3 WorldMax;

		std::vector<ParticleVolumeDescription> ParticleVolumes;
	};

	class SPHSimulation : public RefCounted
	{
	public:
		SPHSimulation(const SPHSimulationDescription& description);
		~SPHSimulation();

		void OnUpdate();

		SPHSimulationDescription GetDescription() const {
			return m_Description;
		}

		SimulationData GetData() const {
			return m_Data;
		}

		const Ref<VertexArray>& GetVAO() {
			return m_PositionVAO[m_CurrentPositionRead];
		}

		const uint32_t GetParticleCount() const {
			return m_Data.ParticleCount;
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

		std::vector<glm::vec4> LoadParticleVolumes() const;
		void SetArray(uint32_t pos, const glm::vec4* data, uint32_t start, uint32_t count);
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

		uint32_t m_CurrentPositionRead;
		uint32_t m_CurrentVelocityRead;
		uint32_t m_CurrentPositionWrite;
		uint32_t m_CurrentVelocityWrite;

		bool m_Initialized = false;

		SPHSimulationDescription m_Description;
		SimulationData m_Data;
	};
}

#endif // !SPH_SIMULATION_H