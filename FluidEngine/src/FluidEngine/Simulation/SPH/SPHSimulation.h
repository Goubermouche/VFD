#ifndef SPH_SIMULATION_H_
#define SPH_SIMULATION_H_

#include "FluidEngine/Simulation/Simulation.h"
#include "FluidEngine/Renderer/Renderer.h"

#include "fluid/pch/header.h"
#include "fluid/CUDA/Params.cuh"
#include "fluid/sph/SPH.h"
#include "fluid/graphics/paramgl.h"


namespace fe {
	class SPHSimulation : public Simulation
	{
	public:
		SPHSimulation();
		~SPHSimulation();

		virtual void OnUpdate() override;
		virtual void OnRender() override;

	private:
		void UpdateEmitter();
	private:
		//  sim
		int emitId, cntRain;
		float3 dyePos;
		SPH* psys;
		float simTime;
		float4 colliderPos;
		float inertia;
		bool paused = false;
		Ref<Material> m_PointMaterial;
	};
}

#endif // !SPH_SIMULATION_H_