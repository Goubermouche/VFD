#ifndef SPH_SIMULATION_H_
#define SPH_SIMULATION_H_

#include "FluidEngine/Simulation/Simulation.h"
#include "FluidEngine/Renderer/Renderer.h"
#include "FluidEngine/Simulation/SPH/cutil/inc/cutil_math.h"
#include "FluidEngine/Simulation/SPH/Params.cuh"
#include "FluidEngine/Compute/GPUCompute.h"

namespace fe {
	class SPHSimulation : public Simulation
	{
	public:
		SPHSimulation();
		~SPHSimulation();

		virtual void OnUpdate() override;
		virtual void OnRender() override;
	private:
		void InitMemory();
		void FreeMemory();

		void UpdateParticles();
		void UpdateGrid();

		void SetArray(bool pos, const float4* data, int start, int count);
		float4* GetArray(bool pos);

		inline  float  length3(float3& a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
		inline  float  length3(float4& a) { return sqrt(a.x * a.x + a.y * a.y + a.z * a.z); }
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

		int* m_Counters;
		int* m_DeltaCounters[2];

		float* m_Pressure;
		float* m_Density;

		Ref<VertexBuffer> m_PositionVBO[2];
		Ref<VertexArray> m_PositionVAO[2];
		Ref<GPUComputeResource> m_Resource[2];

		unsigned int m_CurrentPositionRead;
		unsigned int m_CurrentVelocityRead;
		unsigned int m_CurrentPositionWrite;
		unsigned int m_CurrentVeloctiyWrite;

		bool m_Initialized = false;

		SimParams m_Parameters;

		float m_Spacing;
		float m_CellSize;
		float m_Scale; // only visual

		float3 m_InitMin;
		float3 m_InitMax;

		Ref<Material> m_PointMaterial;
	};
}

#endif // !SPH_SIMULATION_H_