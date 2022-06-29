#ifndef SIM_PARAMS_H_
#define SIM_PARAMS_H_

namespace fe {

#include <vector_types.h>

#define PI   3.141592654f  //3.141592653589793
#define PI2  2.f*PI

	struct SimulationParameters
	{
		float timeStep;
		float globalDamping;
		float particleR;
		float h;
		float h2;
		float SpikyKern;
		float LapKern;
		float Poly6Kern;
		float particleMass;
		float restDensity;
		float stiffness;
		float viscosity;
		float minDens;
		float minDist;
		float distBndHard;
		float distBndSoft;
		float bndDamp;
		float bndStiff;
		float bndDampC;

		unsigned int particleCount;
		unsigned int maxParInCell;
		unsigned int gridSize_yx;
		unsigned int cellCount;

		float3 gravity; 
		float3 cellSize;
		float3 worldMin;
		float3 worldMax;
		float3 worldSize;
		float3 worldMinD;
		float3 worldMaxD;
		float3 worldSizeD;

		uint3 gridSize;
	};
}

#endif // !SIM_PARAMS_H_
