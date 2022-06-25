#include "pch.h"
#include "SPHSimulation.h"



namespace fe {
	SPHSimulation::SPHSimulation()
	{
		/*bInitialized = false;
		hPos = 0; 
		hVel = 0;
		dPos[0] = dPos[1] = 0;	
		dVel[0] = dVel[1] = 0;
		curPosRead = curVelRead = 0;  
		curPosWrite = curVelWrite = 1;*/
	}

	SPHSimulation::~SPHSimulation()
	{
	}

	void SPHSimulation::OnUpdate()
	{
	}

	void SPHSimulation::OnRender()
	{
		Renderer::DrawBox({ 0, 0, 0 }, { 3, 3, 3 }, { 1, 1, 0, 1 });
	}
}