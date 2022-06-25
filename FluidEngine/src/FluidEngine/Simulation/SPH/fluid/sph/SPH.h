#ifndef SPH_H
#define SPH_H

#include "../CUDA/Params.cuh"
#include "../pch/timer.h"
#include "Scene.h"

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

	unsigned int getPosBuffer() const { return posVbo[curPosRead]; };

	unsigned int createVBO(unsigned int size);
	void colorRamp(float t, float* r);
public:		
	float4* hPos, * hVel, * dPos[2], * dVel[2], * dSortedPos, * dSortedVel;
	unsigned int* hParHash, * dParHash[2], * hCellStart, * dCellStart;
	int* hCounters, * dCounters[2];  
	float* dPressure, * dDensity, * dDyeColor;
    unsigned int posVbo[2], colorVbo;
    unsigned int curPosRead, curVelRead, curPosWrite, curVelWrite; 
	Timer tim;	
};

#endif