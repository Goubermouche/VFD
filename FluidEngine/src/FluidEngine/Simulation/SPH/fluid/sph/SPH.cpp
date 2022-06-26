#include "pch.h"

#include "../pch/header.h"
#include "SPH.h"
#include "../CUDA/System.cuh"
#include "../CUDA/radixsort.cuh"
#include "../graphics/param.h"

#pragma region Init
namespace fe {
	SPH::SPH() : bInitialized(false) {

		cudaInitB(0, nullptr, true);
		for (int i = 0; i < NumTim; i++) {
			if (cutCreateTimer(&timer[i]) == CUTFalse) {
				printf("Can't create timer %d !\n", i);
			}
		}

		hPos = 0;
		hVel = 0;
		dPos[0] = dPos[1] = 0;
		dVel[0] = dVel[1] = 0;
		curPosRead = curVelRead = 0;
		curPosWrite = curVelWrite = 1;

		InitScene();
	}

	SPH::~SPH() {
		_FreeMem();
	}

	void SPH::Reset(int type)
	{
		SimParams& p = scn.params;
		float r = p.particleR, spc = scn.spacing, b = p.distBndSoft;
		float3 vr = make_float3(r, r, r), cp = make_float3(p.collPos.x, p.collPos.y, p.collPos.z);
		float3 wMin = p.worldMinD, wSize = p.worldSizeD;
		float3 imin = scn.initMin, imax = scn.initMax;
		float4 pos, vel0 = make_float4(0, 0, 0, 0);
		imin.y += r;	wMin.x += r; wMin.y += r; wMin.z += r;  wSize.x -= 2 * r; wSize.y -= 2 * r; wSize.z -= 2 * r;
		uint i = 0;

		if (type == 1)	//  random
			for (i = 0; i < p.numParticles; ++i)
			{
				hPos[i].x = wMin.x + wSize.x * frand();	hPos[i].y = wMin.y + wSize.y * frand();
				hPos[i].z = wMin.z + wSize.z * frand();	hPos[i].w = 1.f;  hVel[i] = vel0;
			}
		else {		//  volume
#define rst(a)		pos.##a = imin.##a;
#define Inc(a)		pos.##a += spc;  if (pos.##a >= imax.##a)
			rst(x)  rst(y)  rst(z)  pos.w = 1;
#define Inc3(a,b,c)		Inc(a) {  rst(a)  Inc(b) {  rst(b)  Inc(c) rst(c)  }  }
#define INC			switch (scn.initLast)  {  default:\
				case 1:  Inc3(x,z,y);  break;\
				case 0:  Inc3(y,z,x);  break;\
				case 2:  Inc3(x,y,z);  break;	}

			//  pump
			if (p.bndType == BND_PUMP_Y)
			{
				float rad = scn.initMax.x, hc = p.hClose - b * 0.90f, rin = rad * p.radIn - b / 2,
					xs = sinf(p.angOut * p.s3) * rad - b / 2 - p.particleR;

				while (i < p.numParticles)
				{
					if ((p.angOut < 0.5f && pos.y > 0 && pos.z < hc && pos.x < xs && pos.x > -xs) ||	// ^
						(p.angOut >= 0.5f && pos.y > 0 && pos.z < hc && pos.x > 0.09f * p.s4) ||		// S
						sqrt(pos.x * pos.x + pos.y * pos.y) * 1.01f < (pos.z < hc ? rad : rin))
					{
						hPos[i] = pos;	hVel[i] = vel0;  i++;
					}	INC
				}
			}
			else {

				while (i < p.numParticles)
				{
					float3 fff = pos - cp;
					if (length3(fff) > p.collR || p.rotType > 0)  // outside collider
						if (p.bndType == BND_BOX || scn.initType >= 9/**/ || p.bndType == BND_CYL_YZ ||
							(p.bndType == BND_CYL_Y && sqrt(pos.x * pos.x + pos.z * pos.z) < imax.x) ||
							(p.bndType == BND_CYL_Z && sqrt(pos.x * pos.x + pos.y * pos.y) < -imin.y) ||
							(p.bndType == BND_SPHERE && length3(pos) < -imin.y))
						{
							hPos[i] = pos;	hVel[i] = vel0;  i++;
						}	INC
				}
			}
		}
		setArray(0, hPos, 0, p.numParticles);
		setArray(1, hVel, 0, p.numParticles);
	}
	void SPH::_FreeMem()
	{
		if (!bInitialized)  return;  bInitialized = false;

#define  DELA(a)	if (a)	{	delete[]a;  a = NULL;	}

		DELA(hPos);  DELA(hVel);
		DELA(hParHash);  DELA(hCellStart);
		DELA(hCounters);

		freeArray(dVel[0]);		freeArray(dVel[1]);
		freeArray(dSortedPos);	freeArray(dSortedVel);
		freeArray(dPressure);	freeArray(dDensity);
		freeArray(dDyeColor);//

		freeArray(dParHash[0]);  freeArray(dParHash[1]);
		freeArray(dCellStart);
		freeArray(dCounters[0]);  freeArray(dCounters[1]);

		unregGLvbo(posVbo[0]);	unregGLvbo(posVbo[1]);
		unregGLvbo(colorVbo);
		glDeleteBuffers(2, (const GLuint*)posVbo);
		glDeleteBuffers(1, (const GLuint*)&colorVbo);
	}

	void SPH::_InitMem()
	{

		


		if (bInitialized)  return;	bInitialized = true;

		//  CPU data
		uint sflt = sizeof(float), sui = sizeof(uint),
			npar = scn.params.numParticles, ngc = scn.params.numCells,
			memSize1 = npar * sflt, memSize4 = memSize1 * 4;

		hPos = new float4[npar];		memset(hPos, 0, npar * 4 * sflt);
		hVel = new float4[npar];		memset(hVel, 0, npar * 4 * sflt);
		hParHash = new uint[npar * 2];	memset(hParHash, 0, npar * 2 * sui);
		hCellStart = new uint[ngc];	memset(hCellStart, 0, ngc * sui);
		hCounters = new int[10];		memset(hCounters, 0, 10 * sui);


		//  GPU data
		posVbo[0] = createVBO(memSize4);	posVbo[1] = createVBO(memSize4);
		colorVbo = createVBO(memSize4);

		glCreateVertexArrays(1, &posVao[0]);
		glBindVertexArray(0);
		glCreateVertexArrays(1, &posVao[1]);
		glBindVertexArray(0);

		glBindVertexArray(posVao[0]);
		glBindBuffer(GL_ARRAY_BUFFER, posVbo[0]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (const void*)0);

		glBindVertexArray(posVao[1]);
		glBindBuffer(GL_ARRAY_BUFFER, posVbo[1]);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, (const void*)0);

		allocateArray((void**)&dVel[0], memSize4);		
		allocateArray((void**)&dVel[1], memSize4);
		allocateArray((void**)&dSortedPos, memSize4);	
		allocateArray((void**)&dSortedVel, memSize4);
		allocateArray((void**)&dPressure, memSize1);	
		allocateArray((void**)&dDensity, memSize1);
		allocateArray((void**)&dDyeColor, memSize1);//

		allocateArray((void**)&dParHash[0], npar * 2 * sui);	
		allocateArray((void**)&dParHash[1], npar * 2 * sui);
		allocateArray((void**)&dCellStart, ngc * sui);
		allocateArray((void**)&dCounters[0], 100 * sui);		
		allocateArray((void**)&dCounters[1], 100 * sui);

		setParameters(&scn.params);
	}
#pragma endregion

#pragma region Scenes
	void SPH::InitScene()
	{
		_FreeMem();  _InitMem();

		Reset(scn.initType);

		//  app upd   preserve..
		//app::camPosLag = scn.camPos;	app::camRotLag = scn.camRot;
		for (int i = 0; i < NumEmit; i++)
		{
			scn.emit[i].posLag = scn.emit[i].pos;
			scn.emit[i].rotLag = scn.emit[i].rot;
		}
		for (int i = 0; i < NumAcc; i++)
			scn.accPos[i] = scn.params.acc[i].pos;

	}


	void SPH::UpdScene()
	{
		//scn = scenes[curScene];  
		InitScene();
		/*App::updHue();*/
	}

	void SPH::Update()
	{
		if (!bInitialized) {
			printf("return");
			return;
		}
		/*-*/tim.update(true);

		//if (ParamBase::bChangedAny == true)
		//{
		//	ParamBase::bChangedAny = false;
		//	/*if (App::nChg < 20)  App::nChg += 10;*/
		//	// App::updTabVis();/*-*/	App::updBrCnt();
		//	setParameters(&scn.params);
		//	/*tim.update();	t5 = 1000.0*tim.dt;	/**/
		//}


		SimParams& p = scn.params;
		uint2* parHash = (uint2*)dParHash[0];


		///  integrate  (and boundary)
		//**/hCounters[0] = 0;  hCounters[1] = 0;
		//**/copyToDevice(dCounters[curPosWrite], hCounters, 0, 4*sizeof(int));

		cutStartTimer(timer[1]);

		integrate(posVbo[curPosRead], posVbo[curPosWrite],
			dVel[curVelRead], dVel[curVelWrite], p.numParticles/*, dCounters[curPosWrite]*/);

		cutStopTimer(timer[1]);

		//**/copyFromDevice(hCounters, dCounters[curPosRead], 0, 4*sizeof(int));

		swap(curPosRead, curPosWrite);
		swap(curVelRead, curVelWrite);

		// debug -slow
		//copyFromDevice(hPos, 0, posVbo[curPosRead], sizeof(float)*4*50/*p.numParticles*/);
		//copyFromDevice(hVel, 0, colorVbo2/*dVel[curVelRead], 0,*/, sizeof(float)*4*50/*p.numParticles*/);


		///  sort  calculate hash & sort particles
		cutStartTimer(timer[2]);

		calcHash(posVbo[curPosRead], parHash, p.numParticles);
		RadixSort((KeyValuePair*)dParHash[0], (KeyValuePair*)dParHash[1], p.numParticles,
			/*bits*/p.numCells >= 65536 ? 32 : 16);

		cutStopTimer(timer[2]);


		///  reorder particle arrays into sorted order and find start of each cell
		cutStartTimer(timer[3]);

		reorder(posVbo[curPosRead], dVel[curVelRead], dSortedPos, dSortedVel,
			parHash, dCellStart, p.numParticles, p.numCells);

		cutStopTimer(timer[3]);


		///  collisions  (sph density & force)
			cutStartTimer(timer[5]);

		//collide(timer[4], posVbo[curPosRead], posVbo[curPosWrite], /**/colorVbo,
		//	dSortedPos, dSortedVel, dVel[curVelRead], dVel[curVelWrite],
		//	dPressure, dDensity, dDyeColor,//
		//	parHash, dCellStart, p.numParticles, p.numCells);

		cutStopTimer(timer[5]);

		swap(curVelRead, curVelWrite);
	}
#pragma endregion

#pragma region Util
	unsigned int SPH::createVBO(unsigned int size)
	{
		GLuint vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		registerGLvbo(vbo);
		return vbo;
	}

	void SPH::colorRamp(float t, float* r)
	{
		const int ncolors = 4;
		float c[ncolors][3] = {
			{ 0.6f, 0.8f, 1.0f, },
			{ 0.2f, 0.5f, 1.0f, },
			{ 0.5f, 0.2f, 1.0f, },
			{ 1.0f, 1.0f, 1.0f, }, };

		t = t * (ncolors - 1);
		int i = (int)t;
		float u = t - floor(t);

		r[0] = c[i][0] + u * (c[i + 1][0] - c[i][0]);
		r[1] = c[i][1] + u * (c[i + 1][1] - c[i][1]);
		r[2] = c[i][2] + u * (c[i + 1][2] - c[i][2]);
	}

	float4* SPH::getArray(bool pos)
	{
		float4* hdata = 0, * ddata = 0;	uint vbo = 0;
		if (!pos)
		{
			hdata = hPos;  ddata = dPos[curPosRead];	vbo = posVbo[curPosRead];
		}
		else
		{
			hdata = hVel;  ddata = dVel[curVelRead];
		}

		copyFromDevice(hdata, ddata, vbo, scn.params.numParticles * 4 * sizeof(float));
		return hdata;
		return nullptr;
	}


	void SPH::setArray(bool pos, const float4* data, int start, int count)
	{
		const uint si4 = 4 * sizeof(float);
		if (!pos)
		{
			unregGLvbo(posVbo[curPosRead]);
			glBindBuffer(GL_ARRAY_BUFFER, posVbo[curPosRead]);
			glBufferSubData(GL_ARRAY_BUFFER, start * si4, count * si4, data);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
			registerGLvbo(posVbo[curPosRead]);
		}
		else
			copyToDevice(dVel[curVelRead], data, start * si4, count * si4);
	}

}

