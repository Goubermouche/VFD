#ifndef SPH_SIMULATION_H_
#define SPH_SIMULATION_H_

#include "FluidEngine/Simulation/Simulation.h"
#include "FluidEngine/Renderer/Renderer.h"
#include "fluid/pch/header.h"
#include "fluid/CUDA/Params.cuh"
namespace fe {
	class SPHSimulation : public Simulation
	{
	public:
		SPHSimulation();
		~SPHSimulation();

		virtual void OnUpdate() override;
		virtual void OnRender() override;
	public:
	
	};
}

#endif // !SPH_SIMULATION_H_