#ifndef SPH_H
#define SPH_H

#include "../CUDA/Params.cuh"
#include "../pch/timer.h"
#include "FluidEngine/Renderer/Renderer.h"
#include "Scene.h"

static const int NumTim = 6;	// cuda timers

namespace fe {
	class SPH
	{
	public:		///  Methods
		SPH();
		~SPH();

		void _InitMem();
		void _FreeMem();

		bool bInitialized;

		void InitScene();
		void UpdScene();

		void Update();	// simulate
		void Reset(int type);
		Scene scn;
		float4* getArray(bool pos);
		void setArray(bool pos, const float4* data, int start, int count);

		unsigned int createVBO(unsigned int size);
		void colorRamp(float t, float* r);
		Ref<VertexArray> GetPositionVAO() { return positionVAO[curPosRead]; }
	public:
		float4* hPos, * hVel, * dPos[2], * dVel[2], * dSortedPos, * dSortedVel;
		unsigned int* hParHash, * dParHash[2], * hCellStart, * dCellStart;
		int* hCounters, * dCounters[2];
		float* dPressure, * dDensity, * dDyeColor;

		Ref<VertexBuffer> positionVBO[2], colorVBO;
		Ref<VertexArray> positionVAO[2];
		// unsigned int posVbo[2], colorVbo;
		unsigned int curPosRead, curVelRead, curPosWrite, curVelWrite;
		Timer tim;
		unsigned int timer[NumTim];
	};

}

#endif