#ifndef SPH_SIMULATION_H_
#define SPH_SIMULATION_H_

#include "FluidEngine/Simulation/Simulation.h"
#include "FluidEngine/Renderer/Renderer.h"
#include "FluidEngine/Simulation/SPH/cutil/inc/cutil_math.h"

namespace fe {
	class SPHSimulation : public Simulation
	{
	public:
		SPHSimulation();
		~SPHSimulation();

		virtual void OnUpdate() override;
		virtual void OnRender() override;
	private:
		float4* m_Position;
		float4* m_Velocity;
		float4* m_DeltaPosition[2];
		float4* m_DeltaVelocity[2];
		float4* m_SortedPosition;
		float4* m_SortedVelocity;

		unsigned int m_ParticleHash;
		unsigned int m_DeltaParticleHash[2];
		unsigned int m_CellStart;
		unsigned int m_DeltaCellStart;

		int* m_Counters;
		int* m_DeltaCounters[2];

		float* m_Pressure;
		float* m_Density;

		Ref<VertexBuffer> m_PositionVBO;
		Ref<VertexArray> m_PositionVAO;

		unsigned int m_CurrentPositionRead;
		unsigned int m_CurrentVelocityRead;
		unsigned int m_CurrentPositionWrite;
		unsigned int m_CurrentVeloctiyWrite;

		Ref<Material> m_PointMaterial;
	};
}

#endif // !SPH_SIMULATION_H_