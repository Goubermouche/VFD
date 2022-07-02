#include "pch.h"
#include "SPHSimulation.h"
#include "Simulation.cuh"
#include <Glad/glad.h>
#include <FluidEngine/Compute/Utility/CUDAGLInterop.h>
#include "radix/RadixSort.cuh"
#include <FluidEngine/Compute/Utility/cutil.h>

namespace fe {
	SPHSimulation::SPHSimulation()
	{
		m_Position = 0;
		m_Velocity = 0;
		m_DeltaPosition[0] = 0;
		m_DeltaPosition[1] = 0;
		m_DeltaVelocity[0] = 0;
		m_DeltaVelocity[1] = 0;
		m_CurrentPositionRead = 0;
		m_CurrentPositionWrite = 1;
		m_CurrentVelocityRead = 0;
		m_CurrentVeloctiyWrite = 1;

		// Simulation
		unsigned int particleCount = 40000;
		m_Parameters.particleCount = particleCount;
		m_Parameters.maxParInCell = 16;
		m_Parameters.timeStep = 0.0026f;
		m_Parameters.globalDamping = 1.0f;
		m_Parameters.gravity = make_float3(0, -9.81f, 0);

		// SPH
		m_Parameters.particleR = 0.004f;
		m_Parameters.minDist = 1.0f;
		m_Parameters.h = 0.01f;
		m_Spacing = 1.38f;
		m_Parameters.restDensity = 1000;
		m_Parameters.minDens = 1.0f;
		m_Parameters.stiffness = 3.0f;
		m_Parameters.viscosity = 0.5f;

		// World
		float3 w = make_float3(0.2f, 0.25f, 0.2f); // world size
		m_Parameters.worldMin = -w;
		m_Parameters.worldMax = w;
		m_InitMin = -w;
		m_InitMin.y = 0; // TEMP
		m_InitMax = w;
		m_CellSize = m_Parameters.particleR * 2.0f; // = m_Parameters.h
		m_Scale = 10.0f;
		
		// Boundary
		m_Parameters.distBndSoft = 8;
		m_Parameters.distBndHard = 1;
		m_Parameters.bndStiff = 30000;
		m_Parameters.bndDamp = 256;
		m_Parameters.bndDampC = 60;

		UpdateParticles();
		UpdateGrid();

		FreeMemory();
		InitMemory();

		SetParameters(m_Parameters);

		float4 pos;
		float4 float4Zero = make_float4(0, 0, 0, 0);

#define rst(a)		pos.##a = m_InitMin.##a;
#define Inc(a)		pos.##a += m_Spacing;  if (pos.##a >= m_InitMax.##a)
		rst(x)  rst(y)  rst(z)  pos.w = 1;
#define Inc3(a,b,c)		Inc(a) {  rst(a)  Inc(b) {  rst(b)  Inc(c) rst(c)  }  }
#define INC		Inc3(x,z,y);

		for (size_t i = 0; i < particleCount; i++)
		{
			m_Position[i] = pos;
			m_Velocity[i] = float4Zero;
			INC
		}

		SetArray(0, m_Position, 0, m_Parameters.particleCount);
		SetArray(1, m_Velocity, 0, m_Parameters.particleCount);

		// Init material
		m_PointMaterial = Material::Create(Shader::Create("res/Shaders/Normal/PointColorShader.glsl"));
		m_PointMaterial->Set("color", { 0.325,0.455,0.647, 1.0f });
		m_PointMaterial->Set("radius", 0.8f);
		m_PointMaterial->Set("model", glm::scale(glm::mat4(1.0f), { m_Scale, m_Scale, m_Scale }));
	}

	SPHSimulation::~SPHSimulation()
	{
		FreeMemory();
	}

	void SPHSimulation::OnUpdate()
	{
		PROFILE_SCOPE
		if (m_Initialized == false || m_Paused) {
			return;
		}

		uint2* particleHash = (uint2*)m_DeltaParticleHash[0];
		ERR("UPDATE");

		// Integrate
		{
			float4* oldPos;
			float4* newPos;
			GPUCompute::MapResource(m_Resource[m_CurrentPositionRead], (void**)&oldPos);
			GPUCompute::MapResource(m_Resource[m_CurrentPositionWrite], (void**)&newPos);
			Integrate(newPos, m_DeltaVelocity[m_CurrentVeloctiyWrite], oldPos, m_DeltaVelocity[m_CurrentVelocityRead], m_Parameters.particleCount);
			GPUCompute::UnmapResource(m_Resource[m_CurrentPositionRead]);
			GPUCompute::UnmapResource(m_Resource[m_CurrentPositionWrite]);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			WARN("INTEGRATE");
		}

		std::swap(m_CurrentPositionRead, m_CurrentPositionWrite);
		std::swap(m_CurrentVelocityRead, m_CurrentVeloctiyWrite);

		// Hash
		{
			float4* pos;
			GPUCompute::MapResource(m_Resource[m_CurrentPositionRead], (void**)&pos);
			CalculateHash(pos, particleHash, m_Parameters.particleCount);
			GPUCompute::UnmapResource(m_Resource[m_CurrentPositionRead]);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			WARN("HASH");
		}

		// Sort
		{
			RadixSort((KeyValuePair*)m_DeltaParticleHash[0], (KeyValuePair*)m_DeltaParticleHash[1], m_Parameters.particleCount,	m_Parameters.cellCount >= 65536 ? 32 : 16);
			WARN("SORT");
		}

		// Reorder
		{
			float4* oldPos;
			GPUCompute::MapResource(m_Resource[m_CurrentPositionRead], (void**)&oldPos);
			Reorder(particleHash, m_DeltaCellStart, oldPos, m_DeltaVelocity[m_CurrentVelocityRead], m_SortedPosition, m_SortedVelocity, m_Parameters.particleCount, m_Parameters.cellCount);
			GPUCompute::UnmapResource(m_Resource[m_CurrentPositionRead]);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			WARN("REORDER");
		}

		// Collide
		{
			float4* newPos;
			GPUCompute::MapResource(m_Resource[m_CurrentPositionWrite], (void**)&newPos);

			CalculateDensity(m_SortedPosition, m_Pressure, m_Density, particleHash, m_DeltaCellStart, m_Parameters.particleCount);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			WARN("DENSITY");

			CalculateForce(newPos, m_DeltaVelocity[m_CurrentVeloctiyWrite], m_SortedPosition, m_SortedVelocity, m_Pressure, m_Density, particleHash, m_CellStart, m_Parameters.particleCount);
			CUDA_SAFE_CALL(cudaDeviceSynchronize());
			WARN("FORCE");

			GPUCompute::UnmapResource(m_Resource[m_CurrentPositionWrite]);
		}

		std::swap(m_CurrentVelocityRead, m_CurrentVeloctiyWrite);
	}							

	void SPHSimulation::OnRender()
	{
		PROFILE_SCOPE
		float3 worldScale = (m_Parameters.worldMaxD - m_Parameters.worldMinD) * m_Scale;
		Renderer::DrawBox({ 0, 0, 0 }, { worldScale.x, worldScale.y, worldScale.z}, {1.0f, 1.0f, 1.0f, 1.0f});

		// TEMP
		Renderer::DrawPoints(m_PointMaterial);
		m_PositionVAO[m_CurrentPositionRead]->Bind();
		glDrawArrays(GL_POINTS, 0, m_Parameters.particleCount);
	}
	
	void SPHSimulation::InitMemory()
	{
		if (m_Initialized) {
			return;
		}

		// CPU
		unsigned int floatSize = sizeof(float);
		unsigned int uintSize = sizeof(unsigned int);
		unsigned int particleCount = m_Parameters.particleCount;
		unsigned int cellCount = m_Parameters.cellCount;
		unsigned int float1MemorySize = floatSize * particleCount;
		unsigned int float4MemorySize = float1MemorySize * 4;

		m_Position = new float4[particleCount];
		m_Velocity = new float4[particleCount];
		m_ParticleHash = new unsigned int[particleCount * 2];
		m_CellStart = new unsigned int[cellCount];

		memset(m_Position, 0, float4MemorySize);
		memset(m_Velocity, 0, float4MemorySize);
		memset(m_ParticleHash, 0, particleCount * uintSize * 2);
		memset(m_CellStart, 0, cellCount * uintSize);

		// GPU
		m_PositionVAO[0] = VertexArray::Create();
		m_PositionVAO[1] = VertexArray::Create();
		m_PositionVBO[0] = VertexBuffer::Create(float4MemorySize);
		m_PositionVBO[1] = VertexBuffer::Create(float4MemorySize);
		m_PositionVBO[0]->SetLayout({{ShaderDataType::Float4, "a_Position"}});
		m_PositionVBO[1]->SetLayout({{ShaderDataType::Float4, "a_Position"}});
		m_PositionVAO[0]->AddVertexBuffer(m_PositionVBO[0]);
		m_PositionVAO[1]->AddVertexBuffer(m_PositionVBO[1]);
		m_Resource[0] = Ref<GPUComputeResource>::Create();
		m_Resource[1] = Ref<GPUComputeResource>::Create();
		GPUCompute::RegisterBuffer(m_Resource[0], m_PositionVBO[0]); // CHECK
		GPUCompute::RegisterBuffer(m_Resource[1], m_PositionVBO[1]); // CHECK

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_DeltaVelocity[0], float4MemorySize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_DeltaVelocity[1], float4MemorySize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_SortedPosition, float4MemorySize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_SortedVelocity, float4MemorySize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_Pressure, float1MemorySize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_Density, float1MemorySize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_DeltaParticleHash[0], particleCount * 2 * uintSize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_DeltaParticleHash[1], particleCount * 2 * uintSize));
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_DeltaCellStart, cellCount * uintSize));

		m_Initialized = true;

		WARN("memory initialized");
	}

	void SPHSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

#define DELA(a) if(a) { delete[]a; a = NULL; }

		DELA(m_Position);
		DELA(m_Velocity);
		DELA(m_ParticleHash);
		DELA(m_CellStart);

		CUDA_SAFE_CALL(cudaFree(m_DeltaVelocity[0]));
		CUDA_SAFE_CALL(cudaFree(m_DeltaVelocity[1]));
		CUDA_SAFE_CALL(cudaFree(m_SortedPosition));
		CUDA_SAFE_CALL(cudaFree(m_SortedVelocity));
		CUDA_SAFE_CALL(cudaFree(m_Pressure));
		CUDA_SAFE_CALL(cudaFree(m_Density));
		CUDA_SAFE_CALL(cudaFree(m_DeltaParticleHash[0]));
		CUDA_SAFE_CALL(cudaFree(m_DeltaParticleHash[1]));
		CUDA_SAFE_CALL(cudaFree(m_DeltaCellStart));

		WARN("memory freed!");
	}

	void fe::SPHSimulation::UpdateParticles()
	{
		m_Parameters.minDist *= m_Parameters.particleR;
		m_Parameters.h2 = m_Parameters.h * m_Parameters.h;
		m_Spacing *= m_Parameters.particleR;

		m_Parameters.Poly6Kern = 315.0f / (64.0f * PI * pow(m_Parameters.h, 9));
		m_Parameters.SpikyKern = -0.5f * -45.0f / (PI * pow(m_Parameters.h, 6));
		m_Parameters.LapKern = 45.0f / (PI * pow(m_Parameters.h, 6));

		m_Parameters.minDens = 1.0f / pow(m_Parameters.minDens * m_Parameters.restDensity, 2.0f);
		m_Parameters.particleMass = m_Parameters.restDensity * 4.0f / 3.0f * PI * pow(m_Parameters.particleR, 3.0f);

		m_Parameters.distBndSoft *= m_Parameters.particleR;
		m_Parameters.distBndHard *= m_Parameters.particleR;

		WARN("particles updated");
	}

	void fe::SPHSimulation::UpdateGrid()
	{
		float b = m_Parameters.distBndSoft - m_Parameters.particleR;
		float3 b3 = make_float3(b, b, b);

		m_Parameters.worldMinD = m_Parameters.worldMin + b3;
		m_Parameters.worldMaxD = m_Parameters.worldMax - b3;
		m_InitMin += b3;
		m_InitMax -= b3;
		m_Parameters.worldSize = m_Parameters.worldMax - m_Parameters.worldMin;
		m_Parameters.worldSizeD = m_Parameters.worldMaxD - m_Parameters.worldMinD;

		m_Parameters.cellSize = make_float3(m_CellSize, m_CellSize, m_CellSize);
		m_Parameters.gridSize.x = ceil(m_Parameters.worldSize.x / m_Parameters.cellSize.x);
		m_Parameters.gridSize.y = ceil(m_Parameters.worldSize.y / m_Parameters.cellSize.y);
		m_Parameters.gridSize.z = ceil(m_Parameters.worldSize.z / m_Parameters.cellSize.z);
		m_Parameters.gridSize_yx = m_Parameters.gridSize.y * m_Parameters.gridSize.x;
		m_Parameters.cellCount = m_Parameters.gridSize.x * m_Parameters.gridSize.y * m_Parameters.gridSize.z;

		WARN("grid updated");
	}

	// CHECK
	void fe::SPHSimulation::SetArray(bool pos, const float4* data, int start, int count)
	{
		assert(m_Initialized);
		const unsigned int float4MemorySize = 4 * sizeof(float);
		if (pos == false) {
			GPUCompute::UnregisterResource(m_Resource[m_CurrentPositionRead]);
			// cudaGLUnregisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID());
			m_PositionVBO[m_CurrentPositionRead]->SetData(start * float4MemorySize, count * float4MemorySize, data);
			m_PositionVBO[m_CurrentPositionRead]->Unbind();
			GPUCompute::RegisterBuffer(m_Resource[m_CurrentPositionRead], m_PositionVBO[m_CurrentPositionRead]);
			// cudaGLRegisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID());
		}
		else {
			CUDA_SAFE_CALL(cudaMemcpy((float*)m_DeltaVelocity[m_CurrentVelocityRead] + start * float4MemorySize, data, count * float4MemorySize, cudaMemcpyHostToDevice));
		}
	}
}