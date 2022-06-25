#include "pch.h"
#include "SPHSimulation.h"

namespace fe {
	SPHSimulation::SPHSimulation()
	{
		LOG("simulation created!");
	}

	void SPHSimulation::OnUpdate()
	{
	}

	void SPHSimulation::OnRender()
	{
		Renderer::DrawBox({ 0, 0, 0 }, { 3, 3, 3 }, { 1, 1, 0, 1 });
	}
}