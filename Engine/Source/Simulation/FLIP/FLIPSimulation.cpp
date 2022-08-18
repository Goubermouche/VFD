#include "pch.h"
#include "FLIPSimulation.h"

namespace fe {
	FLIPSimulation::FLIPSimulation(const FLIPSimulationDescription& desc)
		: m_Description(desc)
	{
		ERR("FLIP initialized!");
	}

	FLIPSimulation::~FLIPSimulation()
	{
		ERR("FLIP destroyed!");
	}
}