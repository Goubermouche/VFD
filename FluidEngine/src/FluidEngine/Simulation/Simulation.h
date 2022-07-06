#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "pch.h"
#include "FluidEngine/Simulation/ParticleSampler.h"

namespace fe {
	class Simulation : public RefCounted
	{
	public:
		virtual void OnUpdate() = 0;
		virtual void OnRender() = 0;
	};
}

#endif // !SIMULATION_H_