#ifndef SPH_SIMULATION_H_
#define SPH_SIMULATION_H_

#include "FluidEngine/Simulation/Simulation.h"
#include "FluidEngine/Renderer/Renderer.h"

namespace fe {
	class SPHSimulation : public Simulation
	{
	public:
		SPHSimulation();

		virtual void OnUpdate() override;
		virtual void OnRender() override;
	};
}

#endif // !SPH_SIMULATION_H_


