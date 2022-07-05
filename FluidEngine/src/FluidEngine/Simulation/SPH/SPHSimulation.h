#ifndef SPH_SIMULATION_H_
#define SPH_SIMULATION_H_

#include "FluidEngine/Simulation/Simulation.h"
#include "FluidEngine/Renderer/Renderer.h"
#include "FluidEngine/Compute/GPUCompute.h"

#include "SimulationParameters.cuh"

namespace fe {
	class SPHSimulation : public Simulation
	{
	public:
		SPHSimulation();
		~SPHSimulation();

		virtual void OnUpdate() override;
		virtual void OnRender() override;
	private:
		/// <summary>
		/// Sets the initial values for position, velocity, has, and cell start arrays and allocates the neccessary memory.
		/// </summary>
		void InitMemory();
		void FreeMemory();

		/// <summary>
		/// Updates constant particle values based on the current params.
		/// </summary>
		void UpdateParticles();

		/// <summary>
		/// Updates constant grid values based on the current params.
		/// </summary>
		void UpdateGrid();

		void SetArray(bool pos, const float4* data, int start, int count);

		inline float length3(float3& a) { 
			return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); 
		}

		inline float length3(float4& a) {
			return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); 
		}
	private:
		float4* m_Position;
		float4* m_Velocity;
		float4* m_DeltaPosition[2];
		float4* m_DeltaVelocity[2];
		float4* m_SortedPosition;
		float4* m_SortedVelocity;

		unsigned int* m_ParticleHash;
		unsigned int* m_DeltaParticleHash[2];
		unsigned int* m_CellStart;
		unsigned int* m_DeltaCellStart;

		float* m_Pressure;
		float* m_Density;

		Ref<VertexBuffer> m_PositionVBO[2];
		Ref<VertexArray> m_PositionVAO[2];

		unsigned int m_CurrentPositionRead;
		unsigned int m_CurrentVelocityRead;
		unsigned int m_CurrentPositionWrite;
		unsigned int m_CurrentVeloctiyWrite;

		bool m_Initialized = false;
		bool m_Paused = false;

		float m_Spacing;
		float m_CellSize;
		float m_Scale; // only visual

		float3 m_InitMin;
		float3 m_InitMax;

		Ref<Material> m_PointMaterial;
	};
}

#endif // !SPH_SIMULATION_H_