#include "pch.h"
#include "SPHSimulation.h"
#include "Simulation.cuh"

namespace fe {
	const float scale = 10.0f;;
	
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

		// simulation
		m_Parameters.numParticles = 7 * 8 * 1024;
		m_Parameters.maxParInCell = 16;
		m_Parameters.timeStep = 0.0026f;
		m_Parameters.globalDamping = 1.0f;
		m_Parameters.gravity = make_float3(0, -9.81f, 0);

		// sph
		m_Parameters.particleR = 0.004f;
		m_Parameters.minDist = 1.0f;
		m_Parameters.h = 0.01f;
		m_Spacing = 1.38f;
		m_Parameters.restDensity = 1000;
		m_Parameters.minDens = 1.0f;
		m_Parameters.stiffness = 3.0f;
		m_Parameters.viscosity = 0.5f;

		// world
		float3 w = make_float3(0.2f, 0.25f, 0.2f); // world size
		m_Parameters.worldMin = -w;
		m_Parameters.worldMax = w;
		m_InitMin = -w;
		m_InitMax = w;
		m_CellSize = m_Parameters.particleR * 2.0f; // = m_Parameters.h
		
		// boundary
		m_Parameters.distBndSoft = 8;
		m_Parameters.distBndHard = 1;
		m_Parameters.bndStiff = 30000;
		m_Parameters.bndDamp = 256;
		m_Parameters.bndDampC = 60;
		m_Parameters.bndType = BND_BOX;
		m_Parameters.bndEffZ = BND_EFF_NONE;

		FreeMemory();
		InitMemory();

		// Init material
		m_PointMaterial = Material::Create(Shader::Create("res/Shaders/Normal/PointColorShader.glsl"));
		m_PointMaterial->Set("color", { 0.271,1.,0.757, 1 });
		m_PointMaterial->Set("radius", 0.8f);
		m_PointMaterial->Set("model", glm::scale(glm::mat4(1.0f), { scale, scale , scale }));
	}

	SPHSimulation::~SPHSimulation()
	{
	}

	void SPHSimulation::OnUpdate()
	{
	}

	void SPHSimulation::OnRender()
	{
		Renderer::DrawLine({ 0, 0, 0 }, { 10, 10, 10 }, { 1, 1,0,1 });
	}
	
	void SPHSimulation::InitMemory()
	{
		if (m_Initialized) {
			return;
		}

		m_Initialized = true;

		// CPU
		unsigned int floatSize = sizeof(float);
		unsigned int uintSize = sizeof(unsigned int);
		unsigned int particleCount = m_Parameters.numParticles;
		unsigned int cellCount = m_Parameters.numCells;
		unsigned int float1MemorySize = floatSize * particleCount;
		unsigned int float4MemorySize = float1MemorySize * 4;

		m_Position = new float4[particleCount];
		m_Velocity = new float4[particleCount];
		m_ParticleHash = new unsigned int[particleCount * 2];
		m_CellStart = new unsigned int[cellCount];
		m_Counters = new int[10];

		memset(m_Position, 0, float4MemorySize);
		memset(m_Velocity, 0, float4MemorySize);
		memset(m_ParticleHash, 0, particleCount * uintSize * 2);
		memset(m_CellStart, 0, cellCount * uintSize);
		memset(m_Counters, 0, 10 * uintSize);

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

		cudaMalloc((void**)&m_DeltaVelocity[0], float4MemorySize);
		cudaMalloc((void**)&m_DeltaVelocity[1], float4MemorySize);
		cudaMalloc((void**)&m_SortedPosition, float4MemorySize);
		cudaMalloc((void**)&m_SortedVelocity, float4MemorySize);
		cudaMalloc((void**)&m_Pressure, float1MemorySize);
		cudaMalloc((void**)&m_Density, float1MemorySize);
		cudaMalloc((void**)&m_DeltaParticleHash[0], particleCount * uintSize * 2);
		cudaMalloc((void**)&m_DeltaParticleHash[1], particleCount * uintSize * 2);
		cudaMalloc((void**)&m_DeltaCellStart, cellCount * uintSize);
		cudaMalloc((void**)&m_DeltaCounters[0], 100 * uintSize); // CHECK
		cudaMalloc((void**)&m_DeltaCounters[1], 100 * uintSize); // CHECK

		SetParameters(&m_Parameters);

		WARN("memory initialized");
	}

	void SPHSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

#define DELA(a) if(a) {delete[]a; a = NULL;}

		DELA(m_Position);
		DELA(m_Velocity);
		DELA(m_ParticleHash);
		DELA(m_CellStart);
		DELA(m_Counters);

		cudaFree(m_DeltaVelocity[0]);
		cudaFree(m_DeltaVelocity[1]);
		cudaFree(m_SortedPosition);
		cudaFree(m_SortedVelocity);
		cudaFree(m_Pressure);
		cudaFree(m_Density);

		cudaFree(m_DeltaParticleHash[0]);
		cudaFree(m_DeltaParticleHash[1]);
		cudaFree(m_DeltaCellStart);
		cudaFree(m_DeltaCounters[0]);
		cudaFree(m_DeltaCounters[1]);

		WARN("memory freed!");
	}
}