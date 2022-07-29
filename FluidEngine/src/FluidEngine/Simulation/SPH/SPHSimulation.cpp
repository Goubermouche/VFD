#include "pch.h" 
#include "SPHSimulation.h"

#include "FluidEngine/Simulation/SPH/Simulation.cuh"
#include "FluidEngine/Compute/Utility/RadixSort/RadixSort.cuh"

#include <Glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	SPHSimulation::SPHSimulation(const SPHSimulationDescription& description)
		: m_Description(description)
	{
		m_Data.particleRadius = m_Description.particleRadius;
		m_Data.homogenity = m_Description.homogenity;
		m_Data.restDensity = m_Description.restDensity;
		m_Data.stiffness = m_Description.stiffness;
		m_Data.viscosity = m_Description.viscosity;
		m_Data.maxParticlesInCellCount = m_Description.maxParticlesInCellCount;
		m_Data.timeStep = m_Description.timeStep;
		m_Data.globalDamping = m_Description.globalDamping;
		m_Data.gravity = m_Description.gravity;
		m_Data.worldMin = m_Description.worldMin;
		m_Data.worldMax = m_Description.worldMax;
		m_Data.boundsStiffness = m_Description.boundsStiffness;
		m_Data.boundsDamping = m_Description.boundsDamping;
		m_Data.boundsDampingCritical = m_Description.boundsDampingCritical;

		std::vector<glm::vec4> samples = LoadParticleVolumes();

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

		m_Data.particleCount = samples.size();

		UpdateParticles();
		UpdateGrid();

		if (m_Data.particleCount > 0) {
			FreeMemory();
			InitMemory();
		}

		SetParameters(m_Data);

		for (size_t i = 0; i < samples.size(); i++)
		{
			m_Position[i] = samples[i];
		}

		if (m_Data.particleCount > 0) {
			SetArray(0, m_Position, 0, m_Data.particleCount);
			SetArray(1, m_Velocity, 0, m_Data.particleCount);
		}
	
		// Init material
		//m_PointMaterial = Ref < Material>::Create(Ref<Shader>::Create("res/Shaders/Normal/PointDiffuseShader.glsl"));

		//m_PointMaterial->Set("color", { 0.73f, 0.73f, 0.73f, 1.0f });
		//m_PointMaterial->Set("radius", m_Description.particleRadius * 270.0f);
		//m_PointMaterial->Set("model", glm::scale(glm::mat4(1.0f), { 10.0f, 10.0f, 10.0f }));

		LOG("simulation initialized","SPH");
		LOG("samples: " + std::to_string(samples.size()));
		LOG("timestep: " + std::to_string(m_Description.timeStep));
		LOG("viscosity: " + std::to_string(m_Description.viscosity));
	}

	SPHSimulation::~SPHSimulation()
	{
		FreeMemory();
	}

	void SPHSimulation::OnUpdate()
	{
		if (m_Initialized == false || m_Paused) {
			return;
		}

		glm::uvec2* particleHash = (glm::uvec2*)m_DeltaParticleHash[0];

		Integrate(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVeloctiyWrite], m_Data.particleCount);
		std::swap(m_CurrentPositionRead, m_CurrentPositionWrite);
		std::swap(m_CurrentVelocityRead, m_CurrentVeloctiyWrite);
		CalculateHash(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), particleHash, m_Data.particleCount);
		RadixSort((KeyValuePair*)m_DeltaParticleHash[0], (KeyValuePair*)m_DeltaParticleHash[1], m_Data.particleCount, m_Data.cellCount >= 65536 ? 32 : 16);
		Reorder(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_SortedPosition, m_SortedVelocity, particleHash, m_DeltaCellStart, m_Data.particleCount, m_Data.cellCount);
		Collide(m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(),	m_SortedPosition, m_SortedVelocity, m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVeloctiyWrite],	m_Pressure, m_Density, particleHash, m_DeltaCellStart, m_Data.particleCount, m_Data.cellCount);
		std::swap(m_CurrentVelocityRead, m_CurrentVeloctiyWrite);
	}							

	void SPHSimulation::OnRender()
	{
		glm::vec3 worldScale = (m_Data.worldMaxReal - m_Data.worldMinReal) * 10.0f;
		const glm::mat4 mat = glm::scale(glm::mat4(1.0f), { worldScale.x, worldScale.y, worldScale.z });
	}
	
	void SPHSimulation::InitMemory()
	{
		if (m_Initialized) {
			return;
		}

		// CPU
		uint32_t floatSize = sizeof(float);
		uint32_t uintSize = sizeof(uint32_t);
		uint32_t particleCount = m_Data.particleCount;
		uint32_t cellCount = m_Data.cellCount;
		uint32_t float1MemorySize = floatSize * particleCount;
		uint32_t float4MemorySize = float1MemorySize * 4;

		m_Position = new glm::vec4[particleCount];
		m_Velocity = new glm::vec4[particleCount];
		m_ParticleHash = new uint32_t[particleCount * 2];
		m_CellStart = new uint32_t[cellCount];

		memset(m_Position, 0, float4MemorySize);
		memset(m_Velocity, 0, float4MemorySize);
		memset(m_ParticleHash, 0, particleCount * uintSize * 2);
		memset(m_CellStart, 0, cellCount * uintSize);

		// GPU
		m_PositionVAO[0] = Ref<VertexArray>::Create();
		m_PositionVAO[1] = Ref<VertexArray>::Create();
		m_PositionVBO[0] = Ref<VertexBuffer>::Create(float4MemorySize);
		m_PositionVBO[1] = Ref<VertexBuffer>::Create(float4MemorySize);
		m_PositionVBO[0]->SetLayout({{ShaderDataType::Float4, "a_Position"}});
		m_PositionVBO[1]->SetLayout({{ShaderDataType::Float4, "a_Position"}});
		m_PositionVAO[0]->AddVertexBuffer(m_PositionVBO[0]);
		m_PositionVAO[1]->AddVertexBuffer(m_PositionVBO[1]);

		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_PositionVBO[0]->GetRendererID()));
		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_PositionVBO[1]->GetRendererID()));

		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaVelocity[0], float4MemorySize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaVelocity[1], float4MemorySize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_SortedPosition, float4MemorySize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_SortedVelocity, float4MemorySize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_Pressure, float1MemorySize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_Density, float1MemorySize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaParticleHash[0], particleCount * 2 * uintSize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaParticleHash[1], particleCount * 2 * uintSize));
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaCellStart, cellCount * uintSize));

		m_Initialized = true;
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

		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_PositionVBO[0]->GetRendererID()));
		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_PositionVBO[1]->GetRendererID()));

		COMPUTE_SAFE(cudaFree(m_DeltaVelocity[0]));
		COMPUTE_SAFE(cudaFree(m_DeltaVelocity[1]));
		COMPUTE_SAFE(cudaFree(m_SortedPosition));
		COMPUTE_SAFE(cudaFree(m_SortedVelocity));
		COMPUTE_SAFE(cudaFree(m_Pressure));
		COMPUTE_SAFE(cudaFree(m_Density));
		COMPUTE_SAFE(cudaFree(m_DeltaParticleHash[0]));
		COMPUTE_SAFE(cudaFree(m_DeltaParticleHash[1]));
		COMPUTE_SAFE(cudaFree(m_DeltaCellStart));
	}

	void SPHSimulation::UpdateParticles()
	{
		m_Data.minDist = m_Description.particleRadius;
		m_Data.smoothingRadius = m_Description.homogenity * m_Description.homogenity;

		m_Data.poly6Kern = 315.0f / (64.0f * PI * pow(m_Description.homogenity, 9));
		m_Data.spikyKern = -0.5f * -45.0f / (PI * pow(m_Description.homogenity, 6));
		m_Data.lapKern = 45.0f / (PI * pow(m_Description.homogenity, 6));

		m_Data.minDens = 1.0f / pow(m_Description.restDensity, 2.0f);
		m_Data.particleMass = m_Description.restDensity * 4.0f / 3.0f * PI * pow(m_Description.particleRadius, 3.0f);

		m_Data.boundsSoftDistance = 8 * m_Description.particleRadius;
		m_Data.boundsHardDistance = 4 * m_Description.particleRadius;
	}

	void SPHSimulation::UpdateGrid()
	{
		float b = m_Data.boundsSoftDistance - m_Data.particleRadius;
		glm::vec3 b3 = { b, b, b };

		m_Data.worldMinReal = m_Description.worldMin + b3;
		m_Data.worldMaxReal = m_Description.worldMax - b3;
		m_Data.worldSize = m_Description.worldMax - m_Description.worldMin;
		m_Data.worldSizeReal = m_Data.worldMaxReal - m_Data.worldMinReal;

		float cellSize = m_Description.particleRadius * 2.0f;
		m_Data.cellSize = { cellSize, cellSize, cellSize };
		m_Data.gridSize.x = ceil(m_Data.worldSize.x / m_Data.cellSize.x);
		m_Data.gridSize.y = ceil(m_Data.worldSize.y / m_Data.cellSize.y);
		m_Data.gridSize.z = ceil(m_Data.worldSize.z / m_Data.cellSize.z);
		m_Data.gridSizeYX = m_Data.gridSize.y * m_Data.gridSize.x;
		m_Data.cellCount = m_Data.gridSize.x * m_Data.gridSize.y * m_Data.gridSize.z;
	}

	std::vector<glm::vec4> SPHSimulation::LoadParticleVolumes() {
		std::vector<glm::vec4> samples = std::vector<glm::vec4>();

		for (size_t i = 0; i < m_Description.particleVolumes.size(); i++)
		{
			EdgeMesh mesh(m_Description.particleVolumes[i].sourceMesh, m_Description.particleVolumes[i].scale);
			std::vector<glm::vec3> s = ParticleSampler::SampleMesh(mesh, 0.0032f, m_Description.particleVolumes[i].resolution, false, m_Description.particleVolumes[i].sampleMode);

			for (size_t j = 0; j < s.size(); j++)
			{
				samples.push_back({ s[j] + m_Description.particleVolumes[i].position, 0.0f });
			}
		}

		return samples;
	}

	void SPHSimulation::SetArray(bool pos, const glm::vec4* data, int start, int count)
	{
		assert(m_Initialized);
		const uint32_t float4MemorySize = 4 * sizeof(float);
		if (pos == false) {
			COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID()));
			m_PositionVBO[m_CurrentPositionRead]->SetData(start * float4MemorySize, count * float4MemorySize, data);
			m_PositionVBO[m_CurrentPositionRead]->Unbind();
			COMPUTE_SAFE(cudaGLRegisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID()));
		}
		else {
			COMPUTE_SAFE(cudaMemcpy((char*)m_DeltaVelocity[m_CurrentVelocityRead] + start * float4MemorySize, data, count * float4MemorySize, cudaMemcpyHostToDevice));
		}
	}
}