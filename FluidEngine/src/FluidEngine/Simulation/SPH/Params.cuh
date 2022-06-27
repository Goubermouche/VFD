#ifndef SIM_PARAMS_H_
#define SIM_PARAMS_H_

namespace fe {
#ifndef __DEVICE_EMULATION__
#define USE_TEX 1
#endif

#ifdef USE_TEX
	// macro does either global read or texture fetch
#define  FETCH(t,i)  tex1Dfetch(t##Tex, i)
#else
#define  FETCH(t,i)  t[i]
#endif

#include <vector_types.h>

#define PI   3.141592654f  //3.141592653589793
#define PI2  2.f*PI

	enum BndType { BND_BOX = 0, BND_CYL_Y, BND_CYL_Z, BND_CYL_YZ, BND_SPHERE, BND_PUMP_Y, BND_ALL, BND_DW = 0xFFFFffff };
	const static char BndNames[BND_ALL][20] =
	{ "Box", "Cylinder Y", "Cylinder Z", "Cylinders Y,Z", "Sphere", "Pump Y" };

	enum BndEff { BND_EFF_NONE = 0, BND_EFF_WRAP, BND_EFF_CYCLE, BND_EFF_WAVE, BEF_ALL, BEF_DW = 0xFFFFffff };
	const static char BndEffNames[BEF_ALL][20] =
	{ "None", "Wrap Z", "Cycle Z", "Wave Z" };

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

		BndType bndType;
		BndEff bndEffZ;
	};
}

#endif // !SIM_PARAMS_H_
