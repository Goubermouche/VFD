#ifndef DFSPH_DEBUG_INFO_H
#define DFSPH_DEBUG_INFO_H

#include "Debug/Timer.h"

namespace vfd
{
	struct DFSPHDebugInfo
	{
		Timer NeighborhoodSearchTimer;
		Timer BaseSolverTimer;
		Timer DivergenceSolverTimer;
		Timer SurfaceTensionSolverTimer;
		Timer ViscositySolverTimer;
		Timer PressureSolverTimer;

		unsigned int IterationCount = 0u;
		unsigned int DivergenceSolverIterationCount = 0u;
		unsigned int PressureSolverIterationCount = 0u;
		unsigned int ViscositySolverIterationCount = 0u;

		float DivergenceSolverError = 0.0f;
		float PressureSolverError = 0.0f;
		float ViscositySolverError = 0.0f;
	};
}

#endif // !DFSPH_DEBUG_INFO_H