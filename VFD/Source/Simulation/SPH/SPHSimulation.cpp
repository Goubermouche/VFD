#include "pch.h" 
#include "SPHSimulation.h"

#include "Simulation/SPH/SPHSimulation.cuh"
#include "Compute/Utility/RadixSort/RadixSort.cuh"
#include "Debug/SystemInfo.h"
#include "Core/Time.h"

#include <Glad/glad.h>
#include <cuda_gl_interop.h>

namespace vfd {
	SPHSimulation::SPHSimulation(const SPHSimulationDescription& description)
		: m_Description(description)
	{
		if (SystemInfo::CUDADeviceMeetsRequirements() == false) {
			return;
		}

		m_Parameters.ParticleRadius = m_Description.ParticleRadius;
		m_Parameters.Homogeneity = m_Description.Homogeneity;
		m_Parameters.RestDensity = m_Description.RestDensity;
		m_Parameters.Stiffness = m_Description.Stiffness;
		m_Parameters.Viscosity = m_Description.Viscosity;
		m_Parameters.MaxParticlesInCellCount = m_Description.MaxParticlesInCellCount;
		m_Parameters.TimeStep = m_Description.TimeStep;
		m_Parameters.GlobalDamping = m_Description.GlobalDamping;
		m_Parameters.Gravity = m_Description.Gravity;
		m_Parameters.WorldMin = m_Description.WorldMin;
		m_Parameters.WorldMax = m_Description.WorldMax;
		m_Parameters.BoundsStiffness = m_Description.BoundsStiffness;
		m_Parameters.BoundsDamping = m_Description.BoundsDamping;
		m_Parameters.BoundsDampingCritical = m_Description.BoundsDampingCritical;

		m_PositionCache = LoadParticleVolumes();

		m_Position = nullptr;
		m_Velocity = nullptr;
		m_DeltaVelocity[0] = nullptr;
		m_DeltaVelocity[1] = nullptr;
		m_CurrentPositionRead = 0;
		m_CurrentPositionWrite = 1;
		m_CurrentVelocityRead = 0;
		m_CurrentVelocityWrite = 1;

		m_Parameters.ParticleCount = m_PositionCache.size();

		UpdateParticles();
		UpdateGrid();

		if (m_Parameters.ParticleCount > 0) {
			InitMemory();
		}

		SPHUploadSimulationParametersToSymbol(m_Parameters);

		for (uint32_t i = 0; i < m_PositionCache.size(); i++)
		{
			m_Position[i] = m_PositionCache[i];
			m_Velocity[i] = { 0, 0, 0, 0 };
		}

		if (m_Parameters.ParticleCount > 0) {
			SetArray(0, m_Position, 0, m_Parameters.ParticleCount);
			SetArray(1, m_Velocity, 0, m_Parameters.ParticleCount);
		}
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

		for (size_t i = 0; i < m_Description.StepCount; i++)
		{
			const auto particleHash = (glm::uvec2*)m_DeltaParticleHash[0];

			SPHIntegrate(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVelocityWrite], m_Parameters.ParticleCount);
			std::swap(m_CurrentPositionRead, m_CurrentPositionWrite);
			std::swap(m_CurrentVelocityRead, m_CurrentVelocityWrite);
			SPHCalculateHash(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), particleHash, m_Parameters.ParticleCount);
			RadixSort((KeyValuePair*)m_DeltaParticleHash[0], (KeyValuePair*)m_DeltaParticleHash[1], m_Parameters.ParticleCount, m_Parameters.CellCount >= 65536 ? 32 : 16);
			SPHReorder(m_PositionVBO[m_CurrentPositionRead]->GetRendererID(), m_DeltaVelocity[m_CurrentVelocityRead], m_SortedPosition, m_SortedVelocity, particleHash, m_DeltaCellStart, m_Parameters.ParticleCount, m_Parameters.CellCount);
			SPHCollide(m_PositionVBO[m_CurrentPositionWrite]->GetRendererID(), m_SortedPosition, m_SortedVelocity, m_DeltaVelocity[m_CurrentVelocityRead], m_DeltaVelocity[m_CurrentVelocityWrite], m_Pressure, m_Density, particleHash, m_DeltaCellStart, m_Parameters.ParticleCount, m_Parameters.CellCount);
			std::swap(m_CurrentVelocityRead, m_CurrentVelocityWrite);
		}
	}

	void SPHSimulation::Reset()
	{
		FreeMemory();

		if (m_Parameters.ParticleCount > 0) {
			InitMemory();
		}

		for (uint32_t i = 0; i < m_PositionCache.size(); i++)
		{
			m_Position[i] = m_PositionCache[i];
		}

		if (m_Parameters.ParticleCount > 0) {
			SetArray(0, m_Position, 0, m_Parameters.ParticleCount);
			SetArray(1, m_Velocity, 0, m_Parameters.ParticleCount);
		}
	}

	void SPHSimulation::InitMemory()
	{
		// CPU
		constexpr uint32_t floatSize = sizeof(float);
		constexpr uint32_t uintSize = sizeof(uint32_t);
		const uint32_t particleCount = m_Parameters.ParticleCount;
		const uint32_t cellCount = m_Parameters.CellCount;
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

		delete[]m_Position;
		delete[]m_Velocity;
		delete[]m_ParticleHash;
		delete[]m_CellStart;

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

		m_Initialized = false;
	}

	void SPHSimulation::UpdateParticles()
	{
		m_Parameters.MinDist = m_Description.ParticleRadius;
		m_Parameters.SmoothingRadius = m_Description.Homogeneity * m_Description.Homogeneity;

		m_Parameters.Poly6Kern = 315.0f / (64.0f * PI * pow(m_Description.Homogeneity, 9));
		m_Parameters.SpikyKern = -0.5f * -45.0f / (PI * pow(m_Description.Homogeneity, 6));
		m_Parameters.LapKern = 45.0f / (PI * pow(m_Description.Homogeneity, 6));

		m_Parameters.MinDens = 1.0f / pow(m_Description.RestDensity, 2.0f);
		m_Parameters.ParticleMass = m_Description.RestDensity * 4.0f / 3.0f * PI * pow(m_Description.ParticleRadius, 3.0f);

		m_Parameters.BoundsSoftDistance = 8 * m_Description.ParticleRadius;
		m_Parameters.BoundsHardDistance = 4 * m_Description.ParticleRadius;
	}

	void SPHSimulation::UpdateGrid()
	{
		float b = m_Parameters.BoundsSoftDistance - m_Parameters.ParticleRadius;
		const glm::vec3 b3 = { b, b, b };

		m_Parameters.WorldMinReal = m_Description.WorldMin + b3;
		m_Parameters.WorldMaxReal = m_Description.WorldMax - b3;
		m_Parameters.WorldSize = m_Description.WorldMax - m_Description.WorldMin;
		m_Parameters.WorldSizeReal = m_Parameters.WorldMaxReal - m_Parameters.WorldMinReal;

		float cellSize = m_Description.ParticleRadius * 2.0f;
		m_Parameters.CellSize = { cellSize, cellSize, cellSize };
		m_Parameters.GridSize.x = ceil(m_Parameters.WorldSize.x / m_Parameters.CellSize.x);
		m_Parameters.GridSize.y = ceil(m_Parameters.WorldSize.y / m_Parameters.CellSize.y);
		m_Parameters.GridSize.z = ceil(m_Parameters.WorldSize.z / m_Parameters.CellSize.z);
		m_Parameters.GridSizeYX = m_Parameters.GridSize.y * m_Parameters.GridSize.x;
		m_Parameters.CellCount = m_Parameters.GridSize.x * m_Parameters.GridSize.y * m_Parameters.GridSize.z;
	}

	void vfd::SPHSimulation::UpdateDescription(const SPHSimulationDescription& desc)
	{
		m_Description = desc;

		m_Parameters.ParticleRadius = m_Description.ParticleRadius;
		m_Parameters.Homogeneity = m_Description.Homogeneity;
		m_Parameters.RestDensity = m_Description.RestDensity;
		m_Parameters.Stiffness = m_Description.Stiffness;
		m_Parameters.Viscosity = m_Description.Viscosity;
		m_Parameters.MaxParticlesInCellCount = m_Description.MaxParticlesInCellCount;
		m_Parameters.TimeStep = m_Description.TimeStep;
		m_Parameters.GlobalDamping = m_Description.GlobalDamping;
		m_Parameters.Gravity = m_Description.Gravity;
		m_Parameters.WorldMin = m_Description.WorldMin;
		m_Parameters.WorldMax = m_Description.WorldMax;
		m_Parameters.BoundsStiffness = m_Description.BoundsStiffness;
		m_Parameters.BoundsDamping = m_Description.BoundsDamping;
		m_Parameters.BoundsDampingCritical = m_Description.BoundsDampingCritical;

		UpdateParticles();
		UpdateGrid();

		SPHUploadSimulationParametersToSymbol(m_Parameters);
	}

	std::vector<glm::vec4> SPHSimulation::LoadParticleVolumes() const {
		std::vector<glm::vec4> samples = std::vector<glm::vec4>();

		for (const ParticleVolumeDescription& desc : m_Description.ParticleVolumes)
		{
			EdgeMesh mesh(desc.SourceMesh, desc.Scale);

			for (const glm::vec3& sample : ParticleSampler::SampleMeshVolume(mesh, 0.0032f, desc.Resolution, false, desc.SampleMode))
			{
				samples.push_back({ sample + desc.Position, 0.0f });
			}
		}

		return samples;
	}

	void SPHSimulation::SetArray(const uint32_t position, const glm::vec4* data,const uint32_t start,const uint32_t count)
	{
		constexpr uint32_t float4MemorySize = 4 * sizeof(float);
		if (position == false) {
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