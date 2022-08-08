#include "pch.h" 
#include "SPHSimulation.h"

#include "Simulation/SPH/Simulation.cuh"
#include "Compute/Utility/RadixSort/RadixSort.cuh"
#include "Core/Time.h"

#include <Glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	SPHSimulation::SPHSimulation(const SPHSimulationDescription& description)
		: m_Description(description)
	{
		if (GPUCompute::GetInitState() == false) {
			// The GPU compute context failed to initialize. return
			ERR("Simulation stopped (GPU compute context failed to initialize)")
			return;
		}

		m_Data.ParticleRadius = m_Description.ParticleRadius;
		m_Data.Homogeneity = m_Description.Homogeneity;
		m_Data.RestDensity = m_Description.RestDensity;
		m_Data.Stiffness = m_Description.Stiffness;
		m_Data.Viscosity = m_Description.Viscosity;
		m_Data.MaxParticlesInCellCount = m_Description.MaxParticlesInCellCount;
		m_Data.TimeStep = m_Description.TimeStep;
		m_Data.GlobalDamping = m_Description.GlobalDamping;
		m_Data.Gravity = m_Description.Gravity;
		m_Data.WorldMin = m_Description.WorldMin;
		m_Data.WorldMax = m_Description.WorldMax;
		m_Data.BoundsStiffness = m_Description.BoundsStiffness;
		m_Data.BoundsDamping = m_Description.BoundsDamping;
		m_Data.BoundsDampingCritical = m_Description.BoundsDampingCritical;

		std::vector<glm::vec4> samples = LoadParticleVolumes();

		m_Position = nullptr;
		m_Velocity = nullptr;
		m_DeltaVelocity[0] = nullptr;
		m_DeltaVelocity[1] = nullptr;
		m_CurrentPositionRead = 0;
		m_CurrentPositionWrite = 1;
		m_CurrentVelocityRead = 0;
		m_CurrentVelocityWrite = 1;

		m_Data.ParticleCount = samples.size();

		UpdateParticles();
		UpdateGrid();

		if (m_Data.ParticleCount > 0) {
			FreeMemory();
			InitMemory();
		}

		SetParameters(m_Data);

		for (uint32_t i = 0; i < samples.size(); i++)
		{
			m_Position[i] = samples[i];
		}

		if (m_Data.ParticleCount > 0) {
			SetArray(0, m_Position, 0, m_Data.ParticleCount);
			SetArray(1, m_Velocity, 0, m_Data.ParticleCount);
		}

		LOG("simulation initialized", "SPH");
		LOG("samples: " + std::to_string(samples.size()));
		LOG("timestep: " + std::to_string(m_Description.TimeStep));
		LOG("viscosity: " + std::to_string(m_Description.Viscosity));
	}

	SPHSimulation::~SPHSimulation()
	{
		FreeMemory();
	}

	void SPHSimulation::OnUpdate()
	{
		if (m_Initialized == false || paused) {
			return;
		}

		//m_Data.Time += Time::GetDeltaTime();
		//SetParameters(m_Data);

		const auto particleHash = (glm::uvec2*)m_DeltaParticleHash[0];

		Integrate(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVelocityWrite], m_Data.ParticleCount);
		std::swap(m_CurrentPositionRead, m_CurrentPositionWrite);
		std::swap(m_CurrentVelocityRead, m_CurrentVelocityWrite);
		CalculateHash(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), particleHash, m_Data.ParticleCount);
		RadixSort((KeyValuePair*)m_DeltaParticleHash[0], (KeyValuePair*)m_DeltaParticleHash[1], m_Data.ParticleCount, m_Data.CellCount >= 65536 ? 32 : 16);
		Reorder(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_SortedPosition, m_SortedVelocity, particleHash, m_DeltaCellStart, m_Data.ParticleCount, m_Data.CellCount);
		Collide(m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(), m_SortedPosition, m_SortedVelocity, m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVelocityWrite], m_Pressure, m_Density, particleHash, m_DeltaCellStart, m_Data.ParticleCount, m_Data.CellCount);
		std::swap(m_CurrentVelocityRead, m_CurrentVelocityWrite);
	}

	void SPHSimulation::InitMemory()
	{
		if (m_Initialized) {
			return;
		}

		// CPU
		constexpr uint32_t floatSize = sizeof(float);
		constexpr uint32_t uintSize = sizeof(uint32_t);
		const uint32_t particleCount = m_Data.ParticleCount;
		const uint32_t cellCount = m_Data.CellCount;
		const uint32_t float1MemorySize = floatSize * particleCount;
		const uint32_t float4MemorySize = float1MemorySize * 4;

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
		m_PositionVBO[0]->SetLayout({ {ShaderDataType::Float4, "a_Position"} });
		m_PositionVBO[1]->SetLayout({ {ShaderDataType::Float4, "a_Position"} });
		m_PositionVAO[0]->AddVertexBuffer(m_PositionVBO[0]);
		m_PositionVAO[1]->AddVertexBuffer(m_PositionVBO[1]);

		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_PositionVBO[0]->GetRendererID()))
		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_PositionVBO[1]->GetRendererID()))

		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaVelocity[0], float4MemorySize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaVelocity[1], float4MemorySize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_SortedPosition, float4MemorySize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_SortedVelocity, float4MemorySize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_Pressure, float1MemorySize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_Density, float1MemorySize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaParticleHash[0], particleCount * 2 * uintSize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaParticleHash[1], particleCount * 2 * uintSize))
		COMPUTE_SAFE(cudaMalloc((void**)&m_DeltaCellStart, cellCount * uintSize))

		m_Initialized = true;
	}

	void SPHSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

		delete[]m_Position;
		delete[]m_Velocity;
		delete[]m_ParticleHash;
		delete[]m_CellStart;

		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_PositionVBO[0]->GetRendererID()));
		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_PositionVBO[1]->GetRendererID()));

		COMPUTE_SAFE(cudaFree(m_DeltaVelocity[0]))
		COMPUTE_SAFE(cudaFree(m_DeltaVelocity[1]))
		COMPUTE_SAFE(cudaFree(m_SortedPosition))
		COMPUTE_SAFE(cudaFree(m_SortedVelocity))
		COMPUTE_SAFE(cudaFree(m_Pressure))
		COMPUTE_SAFE(cudaFree(m_Density))
		COMPUTE_SAFE(cudaFree(m_DeltaParticleHash[0]))
		COMPUTE_SAFE(cudaFree(m_DeltaParticleHash[1]))
		COMPUTE_SAFE(cudaFree(m_DeltaCellStart))
	}

	void SPHSimulation::UpdateParticles()
	{
		m_Data.MinDist = m_Description.ParticleRadius;
		m_Data.SmoothingRadius = m_Description.Homogeneity * m_Description.Homogeneity;

		m_Data.Poly6Kern = 315.0f / (64.0f * PI * pow(m_Description.Homogeneity, 9));
		m_Data.SpikyKern = -0.5f * -45.0f / (PI * pow(m_Description.Homogeneity, 6));
		m_Data.LapKern = 45.0f / (PI * pow(m_Description.Homogeneity, 6));

		m_Data.MinDens = 1.0f / pow(m_Description.RestDensity, 2.0f);
		m_Data.ParticleMass = m_Description.RestDensity * 4.0f / 3.0f * PI * pow(m_Description.ParticleRadius, 3.0f);

		m_Data.BoundsSoftDistance = 8 * m_Description.ParticleRadius;
		m_Data.BoundsHardDistance = 4 * m_Description.ParticleRadius;
	}

	void SPHSimulation::UpdateGrid()
	{
		float b = m_Data.BoundsSoftDistance - m_Data.ParticleRadius;
		const glm::vec3 b3 = { b, b, b };

		m_Data.WorldMinReal = m_Description.WorldMin + b3;
		m_Data.WorldMaxReal = m_Description.WorldMax - b3;
		m_Data.WorldSize = m_Description.WorldMax - m_Description.WorldMin;
		m_Data.WorldSizeReal = m_Data.WorldMaxReal - m_Data.WorldMinReal;

		float cellSize = m_Description.ParticleRadius * 2.0f;
		m_Data.CellSize = { cellSize, cellSize, cellSize };
		m_Data.GridSize.x = ceil(m_Data.WorldSize.x / m_Data.CellSize.x);
		m_Data.GridSize.y = ceil(m_Data.WorldSize.y / m_Data.CellSize.y);
		m_Data.GridSize.z = ceil(m_Data.WorldSize.z / m_Data.CellSize.z);
		m_Data.GridSizeYX = m_Data.GridSize.y * m_Data.GridSize.x;
		m_Data.CellCount = m_Data.GridSize.x * m_Data.GridSize.y * m_Data.GridSize.z;
	}

	std::vector<glm::vec4> SPHSimulation::LoadParticleVolumes() const {
		std::vector<glm::vec4> samples = std::vector<glm::vec4>();

		for (uint16_t i = 0; i < m_Description.ParticleVolumes.size(); i++)
		{
			EdgeMesh mesh(m_Description.ParticleVolumes[i].SourceMesh, m_Description.ParticleVolumes[i].Scale);
			std::vector<glm::vec3> s = ParticleSampler::SampleMeshVolume(mesh, 0.0032f, m_Description.ParticleVolumes[i].Resolution, false, m_Description.ParticleVolumes[i].SampleMode);

			for (uint32_t j = 0; j < s.size(); j++)
			{
				samples.push_back({ s[j] + m_Description.ParticleVolumes[i].Position, 0.0f });
			}
		}

		return samples;
	}

	void SPHSimulation::SetArray(const uint32_t position, const glm::vec4* data,const uint32_t start,const uint32_t count)
	{
		assert(m_Initialized);
		constexpr uint32_t float4MemorySize = 4 * sizeof(float);
		if (position == false) {
			COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID()))
			m_PositionVBO[m_CurrentPositionRead]->SetData(start * float4MemorySize, count * float4MemorySize, data);
			m_PositionVBO[m_CurrentPositionRead]->Unbind();
			COMPUTE_SAFE(cudaGLRegisterBufferObject(m_PositionVBO[m_CurrentPositionRead]->GetRendererID()))
		}
		else {
			COMPUTE_SAFE(cudaMemcpy((char*)m_DeltaVelocity[m_CurrentVelocityRead] + start * float4MemorySize, data, count * float4MemorySize, cudaMemcpyHostToDevice))
		}
	}
}