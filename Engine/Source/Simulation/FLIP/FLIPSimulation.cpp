#include "pch.h"
#include "FLIPSimulation.h"

#include "Simulation/FLIP/Simulation.cuh"

namespace fe {
	FLIPSimulation::FLIPSimulation(const FLIPSimulationDescription& desc)
		: m_Description(desc)
	{
	}

	FLIPSimulation::~FLIPSimulation()
	{
	}

	void FLIPSimulation::OnUpdate()
	{
	}

	void FLIPSimulation::InitMemory()
	{
	}

	void FLIPSimulation::FreeMemory()
	{
	}
}