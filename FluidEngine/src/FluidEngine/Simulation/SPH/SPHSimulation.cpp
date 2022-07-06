#include "pch.h"
#include "SPHSimulation.h"

#include "FluidEngine/Simulation/SPH/Simulation.cuh"
#include "FluidEngine/Compute/Utility/RadixSort/RadixSort.cuh"
#include "FluidEngine/Compute/Utility/CUDA/cutil.h"
#include "FluidEngine/Compute/Utility/CUDA/cutil_math.h"
#include "FluidEngine/Core/Math/Math.h"
#include "FluidEngine/Core/Time.h"

#define TINYOBJLOADER_IMPLEMENTATION
// #define TINYOBJLOADER_USE_MAPBOX_EARCUT
#include "tiny_obj_loader.h"

#include <Glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	SimulationParameters m_Parameters;

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

		// SPH
		m_Parameters.particleRadius = 0.004f;
		m_Parameters.minDist = 1.0f;
		m_Parameters.homogenity = 0.01f;
		m_Spacing = 1.38f;
		m_Parameters.restDensity = 1000;
		m_Parameters.minDens = 1.0f;
		m_Parameters.stiffness = 3.0f;
		m_Parameters.viscosity = 0.5f;

		// Sampling
		// TODO: create a separate model loading class
		tinyobj::ObjReaderConfig reader_config;
		tinyobj::ObjReader reader;

		if (!reader.ParseFromFile("res/Models/bunny.obj", reader_config)) {
			if (!reader.Error().empty()) {
				std::cerr << "TinyObjReader: " << reader.Error();
			}
			exit(1);
		}

		if (!reader.Warning().empty()) {
			std::cout << "TinyObjReader: " << reader.Warning();
		}



		auto& attrib = reader.GetAttrib();
		auto& shapes = reader.GetShapes();
		auto& materials = reader.GetMaterials();


		// Simulation
		unsigned int particleCount = 130000;
		m_Parameters.particleCount = particleCount;
		m_Parameters.maxParticlesInCellCount = 32; // higher value increases stability
		m_Parameters.timeStep = 0.0016f;
		m_Parameters.globalDamping = 1.0f;
		m_Parameters.gravity = make_float3(0, -9.81f, 0);



		// World
		float3 w = make_float3(0.4f); // world size
		m_Parameters.worldMin = -w;
		m_Parameters.worldMax = w;
		m_InitMin = -w;
		m_InitMin.y = 0; // TEMP
		m_InitMax = w;
		m_CellSize = m_Parameters.particleRadius * 2.0f;
		m_Scale = 10.0f;
		
		// Boundary
		m_Parameters.boundsSoftDistance = 8;
		m_Parameters.boundsHardDistance = 4;
		m_Parameters.boundsStiffness = 65536;
		m_Parameters.boundsDamping = 256;
		m_Parameters.boundsDampingCritical = 60;

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

		m_PointMaterial->Set("color", { 0.65f, 0.7f, 0.7f, 1.0f });
		m_PointMaterial->Set("radius", m_Parameters.particleRadius * 250.0f);
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

		Integrate(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVeloctiyWrite], m_Parameters.particleCount);
		std::swap(m_CurrentPositionRead, m_CurrentPositionWrite);
		std::swap(m_CurrentVelocityRead, m_CurrentVeloctiyWrite);
		CalculateHash(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), particleHash, m_Parameters.particleCount);
		RadixSort((KeyValuePair*)m_DeltaParticleHash[0], (KeyValuePair*)m_DeltaParticleHash[1], m_Parameters.particleCount,	m_Parameters.cellCount >= 65536 ? 32 : 16);
		Reorder(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_SortedPosition, m_SortedVelocity, particleHash, m_DeltaCellStart, m_Parameters.particleCount, m_Parameters.cellCount);
		Collide(m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(),	m_SortedPosition, m_SortedVelocity, m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVeloctiyWrite],	m_Pressure, m_Density, particleHash, m_DeltaCellStart, m_Parameters.particleCount, m_Parameters.cellCount);
		std::swap(m_CurrentVelocityRead, m_CurrentVeloctiyWrite);
	}							

	void SPHSimulation::OnRender()
	{
		PROFILE_SCOPE
		float3 worldScale = (m_Parameters.worldMaxReal - m_Parameters.worldMinReal) * m_Scale;
		Renderer::DrawBox({ 0, 0, 0 }, { worldScale.x, worldScale.y, worldScale.z}, {1.0f, 1.0f, 1.0f, 1.0f});

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

		CUDA_SAFE_CALL(cudaGLRegisterBufferObject(m_PositionVBO[0]->GetRendererID()));
		CUDA_SAFE_CALL(cudaGLRegisterBufferObject(m_PositionVBO[1]->GetRendererID()));

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
		m_Parameters.minDist *= m_Parameters.particleRadius;
		m_Parameters.smoothingRadius = m_Parameters.homogenity * m_Parameters.homogenity;
		m_Spacing *= m_Parameters.particleRadius;

		m_Parameters.poly6Kern = 315.0f / (64.0f * PI * pow(m_Parameters.homogenity, 9));
		m_Parameters.spikyKern = -0.5f * -45.0f / (PI * pow(m_Parameters.homogenity, 6));
		m_Parameters.lapKern = 45.0f / (PI * pow(m_Parameters.homogenity, 6));

		m_Parameters.minDens = 1.0f / pow(m_Parameters.minDens * m_Parameters.restDensity, 2.0f);
		m_Parameters.particleMass = m_Parameters.restDensity * 4.0f / 3.0f * PI * pow(m_Parameters.particleRadius, 3.0f);

		m_Parameters.boundsSoftDistance *= m_Parameters.particleRadius;
		m_Parameters.boundsHardDistance *= m_Parameters.particleRadius;

		WARN("particles updated");
	}

	void fe::SPHSimulation::UpdateGrid()
	{
		float b = m_Parameters.boundsSoftDistance - m_Parameters.particleRadius;
		float3 b3 = make_float3(b, b, b);

		m_Parameters.worldMinReal = m_Parameters.worldMin + b3;
		m_Parameters.worldMaxReal = m_Parameters.worldMax - b3;
		m_InitMin += b3;
		m_InitMax -= b3;
		m_Parameters.worldSize = m_Parameters.worldMax - m_Parameters.worldMin;
		m_Parameters.worldSizeReal = m_Parameters.worldMaxReal - m_Parameters.worldMinReal;

		m_Parameters.cellSize = make_float3(m_CellSize, m_CellSize, m_CellSize);
		m_Parameters.gridSize.x = ceil(m_Parameters.worldSize.x / m_Parameters.cellSize.x);
		m_Parameters.gridSize.y = ceil(m_Parameters.worldSize.y / m_Parameters.cellSize.y);
		m_Parameters.gridSize.z = ceil(m_Parameters.worldSize.z / m_Parameters.cellSize.z);
		m_Parameters.gridSizeYX = m_Parameters.gridSize.y * m_Parameters.gridSize.x;
		m_Parameters.cellCount = m_Parameters.gridSize.x * m_Parameters.gridSize.y * m_Parameters.gridSize.z;

		WARN("grid updated");
	}

	// CHECK
	void fe::SPHSimulation::SetArray(bool pos, const float4* data, int start, int count)
	{
		assert(m_Initialized);
		const unsigned int float4MemorySize = 4 * sizeof(float);
		if (pos == false) {
			CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID()));
			m_PositionVBO[m_CurrentPositionRead]->SetData(start * float4MemorySize, count * float4MemorySize, data);
			m_PositionVBO[m_CurrentPositionRead]->Unbind();
			CUDA_SAFE_CALL(cudaGLRegisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID()));
		}
		else {
			CUDA_SAFE_CALL(cudaMemcpy((char*)m_DeltaVelocity[m_CurrentVelocityRead] + start * float4MemorySize, data, count * float4MemorySize, cudaMemcpyHostToDevice));
		}
	}
}