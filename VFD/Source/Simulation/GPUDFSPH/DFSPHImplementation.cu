#include "pch.h"

#include "DFSPHImplementation.h"

#include <cuda_gl_interop.h>
#include "Compute/ComputeHelper.h"
#include "Core/Math/Math.h"

#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <functional>

namespace vfd
{
	DFSPHImplementation::DFSPHImplementation(const GPUDFSPHSimulationDescription& desc, std::vector<Ref<RigidBody>>& rigidBodies)
		 : m_Description(desc)
	{
		InitFluidData();
		InitRigidBodies(rigidBodies);

		// Neighborhood search
		m_NeighborhoodSearch = new NeighborhoodSearch(m_Info.SupportRadius);
		m_NeighborhoodSearch->AddPointSet(m_Particles, m_Info.ParticleCount);
	}

	DFSPHImplementation::~DFSPHImplementation()
	{
		delete m_NeighborhoodSearch;
		delete[] m_Particles;
		delete[] m_Particles0;

		COMPUTE_SAFE(cudaFree(d_Info))
		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_VertexBuffer->GetRendererID()))

		// Free rigid body data (only device-side)
		// TODO: implement Free() destructor functions for rigid bodies
		COMPUTE_SAFE(cudaFree(m_RigidBodyPointerWrapper->Nodes))
		COMPUTE_SAFE(cudaFree(m_RigidBodyPointerWrapper->CellMap))
		COMPUTE_SAFE(cudaFree(m_RigidBodyPointerWrapper->Cells))
		COMPUTE_SAFE(cudaFree(m_RigidBodyPointerWrapper->BoundaryXJ))
		COMPUTE_SAFE(cudaFree(m_RigidBodyPointerWrapper->BoundaryVolume))
		COMPUTE_SAFE(cudaFree(m_RigidBodyPointerWrapper->RigidBody))
		delete m_RigidBodyPointerWrapper;
	}

	void DFSPHImplementation::OnUpdate()
	{
		if (m_Info.ParticleCount == 0) {
			return;
		}

		// Map OpenGL memory to CUDA memory
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))

		// Sort all particles based on their radius and position
		m_NeighborhoodSearch->FindNeighbors();
		if (m_IterationCount % 500 == 0) {
			PointSet& pointSet = m_NeighborhoodSearch->GetPointSet(0);
			pointSet.SortField(particles);
		}

		// Simulate
		{
			// Compute boundaries
			ComputeVolumeAndBoundaryKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, m_RigidBodyPointerWrapper->RigidBody);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Clear accelerations
			ClearAccelerationsKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Update time step size
			CalculateTimeStepSize(thrust::device_pointer_cast(particles));
		
			// Calculate velocities
			CalculateVelocitiesKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Calculate positions
			CalculatePositionsKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())
		}

		// Unmap OpenGL memory 
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		// Debug, after the offline solution gets properly implemented this function only needs to be called once
		// after the simulation finishes baking.
		// COMPUTE_SAFE(cudaMemcpy(&m_Info, d_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyDeviceToHost))

		m_IterationCount++;
	}

	void DFSPHImplementation::Reset()
	{
		// Reset particle positions and velocity
		for (size_t i = 0; i < m_Info.ParticleCount; i++)
		{
			DFSPHParticle particle{};

			// Particle data
			particle.Position = m_Particles0[i].Position;
			particle.Velocity = m_Particles0[i].Velocity;
			particle.Acceleration = { 0.0f, 0.0f, 0.0f };
			particle.Mass = m_Info.Volume * m_Info.Density0;
			particle.Density = m_Info.Density0;
			particle.Kappa = 0.0f;
			particle.KappaVelocity = 0.0f;

			// Viscosity
			particle.ViscosityDifference = { 0.0f, 0.0f, 0.0f };

			// Surface tension
			particle.MonteCarloSurfaceNormals = { 0.0f, 0.0f, 0.0f };
			particle.MonteCarloSurfaceNormalsSmooth = { 0.0f, 0.0f, 0.0f };
			particle.FinalCurvature = 0.0f;
			particle.DeltaFinalCurvature = 0.0f;
			particle.SmoothedCurvature = 0.0f;
			particle.MonteCarloSurfaceCurvature = 0.0f;
			particle.MonteCarloSurfaceCurvatureSmooth = 0.0f;
			particle.ClassifierInput = 0.0f;
			particle.ClassifierOutput = 0.0f;

			m_Particles[i] = particle;
		}

		m_VertexBuffer->SetData(0, m_Info.ParticleCount * sizeof(DFSPHParticle), m_Particles);
		m_VertexBuffer->Unbind();

		// Map OpenGL memory to CUDA memory
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))

		CalculateMaxVelocityMagnitude(thrust::device_pointer_cast(particles), 0.0f);

		// Unmap OpenGL memory 
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		m_IterationCount = 0;

		// Reset the time step size
		m_Info.TimeStepSize = m_Description.TimeStepSize;
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))
	}

	const Ref<VertexArray>& DFSPHImplementation::GetVertexArray() const
	{
		return m_VertexArray;
	}

	unsigned int DFSPHImplementation::GetParticleCount() const
	{
		return m_Info.ParticleCount;
	}

	float DFSPHImplementation::GetMaxVelocityMagnitude() const
	{
		return m_MaxVelocityMagnitude;
	}

	float DFSPHImplementation::GetTimeStepSize() const
	{
		return m_Info.TimeStepSize;
	}

	void DFSPHImplementation::InitRigidBodies(std::vector<Ref<RigidBody>>& rigidBodies)
	{
		// Copies the flat rigid body structure to the device
		// Right now I only copy 1 rigid body for the purposes of testing
		// TODO: add support for more rigid bodies
		// TODO: add a CopyToDevice() function to the rigid body class
		// TODO: add a function for copying member arrays

		m_RigidBodyPointerWrapper = new RigidBodyDeviceData();
		RigidBodyData* data = rigidBodies[0]->GetData();

		data->BoundaryXJ = new glm::vec3[m_Info.ParticleCount]();
		data->BoundaryVolume = new float[m_Info.ParticleCount]();

		// Copy the rigid body itself to the device
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&m_RigidBodyPointerWrapper->RigidBody), sizeof(RigidBodyData)))
		COMPUTE_SAFE(cudaMemcpy(m_RigidBodyPointerWrapper->RigidBody, data, sizeof(RigidBodyData), cudaMemcpyHostToDevice))

		// Copy the nodes over to the device
		const unsigned long long int nodesSize = static_cast<unsigned long long>(data->NodeCount) * data->NodeElementCount * sizeof(double);
		COMPUTE_SAFE(cudaMalloc(&m_RigidBodyPointerWrapper->Nodes, nodesSize))
		COMPUTE_SAFE(cudaMemcpy(m_RigidBodyPointerWrapper->Nodes, data->Nodes, nodesSize, cudaMemcpyHostToDevice))
		COMPUTE_SAFE(cudaMemcpy(&m_RigidBodyPointerWrapper->RigidBody->Nodes, &m_RigidBodyPointerWrapper->Nodes, sizeof(double*), cudaMemcpyHostToDevice))

		// Copy the cell map over to the device
		const unsigned long long int cellMapSize = static_cast<unsigned long long>(data->CellMapCount) * data->CellMapElementCount * sizeof(unsigned int);
		COMPUTE_SAFE(cudaMalloc(&m_RigidBodyPointerWrapper->CellMap, cellMapSize))
		COMPUTE_SAFE(cudaMemcpy(m_RigidBodyPointerWrapper->CellMap, data->CellMap, cellMapSize, cudaMemcpyHostToDevice))
		COMPUTE_SAFE(cudaMemcpy(&m_RigidBodyPointerWrapper->RigidBody->CellMap, &m_RigidBodyPointerWrapper->CellMap, sizeof(unsigned int*), cudaMemcpyHostToDevice))

		// Copy the cells over to the device
		const unsigned long long int cellsSize = data->CellCount * data->CellElementCount * 32u * sizeof(unsigned int);
		COMPUTE_SAFE(cudaMalloc(&m_RigidBodyPointerWrapper->Cells, cellsSize))
		COMPUTE_SAFE(cudaMemcpy(m_RigidBodyPointerWrapper->Cells, data->Cells, cellsSize, cudaMemcpyHostToDevice))
		COMPUTE_SAFE(cudaMemcpy(&m_RigidBodyPointerWrapper->RigidBody->Cells, &m_RigidBodyPointerWrapper->Cells, sizeof(unsigned int*), cudaMemcpyHostToDevice))

		// Copy the boundary XJ over to the device
		const unsigned int boundaryXJSize = m_Info.ParticleCount * sizeof(glm::vec3);
		COMPUTE_SAFE(cudaMalloc(&m_RigidBodyPointerWrapper->BoundaryXJ, boundaryXJSize))
		COMPUTE_SAFE(cudaMemcpy(m_RigidBodyPointerWrapper->BoundaryXJ, data->BoundaryXJ, boundaryXJSize, cudaMemcpyHostToDevice))
		COMPUTE_SAFE(cudaMemcpy(&m_RigidBodyPointerWrapper->RigidBody->BoundaryXJ, &m_RigidBodyPointerWrapper->BoundaryXJ, sizeof(glm::vec3*), cudaMemcpyHostToDevice))

		// Copy the boundary volume over to the device
		const unsigned int boundaryVolumeSize = m_Info.ParticleCount * sizeof(float);
		COMPUTE_SAFE(cudaMalloc(&m_RigidBodyPointerWrapper->BoundaryVolume, boundaryVolumeSize))
		COMPUTE_SAFE(cudaMemcpy(m_RigidBodyPointerWrapper->BoundaryVolume, data->BoundaryVolume, boundaryVolumeSize, cudaMemcpyHostToDevice))
		COMPUTE_SAFE(cudaMemcpy(&m_RigidBodyPointerWrapper->RigidBody->BoundaryVolume, &m_RigidBodyPointerWrapper->BoundaryVolume, sizeof(float*), cudaMemcpyHostToDevice))
	}

	void DFSPHImplementation::InitFluidData()
	{
		const glm::vec3 boxPosition = { 0.0f, 2.0f, 0.0f };
		const glm::uvec3 boxSize = { 20, 20, 20 };

		glm::vec3 boxHalfSize = static_cast<glm::vec3>(boxSize - glm::uvec3(1)) / 2.0f;
		unsigned int boxIndex = 0;

		m_Info.ParticleCount = glm::compMul(boxSize);
		m_Info.ParticleRadius = m_Description.ParticleRadius;
		m_Info.ParticleDiameter = 2.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius = 4.0f * m_Info.ParticleRadius;
		m_Info.TimeStepSize = m_Description.TimeStepSize;
		m_Info.Volume = 0.8f * m_Info.ParticleDiameter * m_Info.ParticleDiameter * m_Info.ParticleDiameter;
		m_Info.Density0 = 1000.0f;
		m_Info.WZero = 0.0f;
		m_Info.Gravity = m_Description.Gravity;

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_Info), sizeof(DFSPHSimulationInfo)))
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))

		m_Particles = new DFSPHParticle[m_Info.ParticleCount];
		m_Particles0 = new DFSPHParticle0[m_Info.ParticleCount];

		// Generate a simple box for the purposes of testing 
		for (unsigned int x = 0u; x < boxSize.x; x++)
		{
			for (unsigned int y = 0u; y < boxSize.y; y++)
			{
				for (unsigned int z = 0u; z < boxSize.z; z++)
				{
					DFSPHParticle particle{};

					// Particle data
					particle.Position = (static_cast<glm::vec3>(glm::uvec3(x, y, z)) - boxHalfSize) * m_Info.ParticleDiameter + boxPosition;
					particle.Velocity = { 0.0f, 0.0f, 0.0f };
					particle.Acceleration = { 0.0f, 0.0f, 0.0f };
					particle.Mass = m_Info.Volume * m_Info.Density0;
					particle.Density = m_Info.Density0;
					particle.Kappa = 0.0f;
					particle.KappaVelocity = 0.0f;

					// Viscosity
					particle.ViscosityDifference = { 0.0f, 0.0f, 0.0f };

					// Surface tension
					particle.MonteCarloSurfaceNormals = { 0.0f, 0.0f, 0.0f };
					particle.MonteCarloSurfaceNormalsSmooth = { 0.0f, 0.0f, 0.0f };
					particle.FinalCurvature = 0.0f;
					particle.DeltaFinalCurvature = 0.0f;
					particle.SmoothedCurvature = 0.0f;
					particle.MonteCarloSurfaceCurvature = 0.0f;
					particle.MonteCarloSurfaceCurvatureSmooth = 0.0f;
					particle.ClassifierInput = 0.0f;
					particle.ClassifierOutput = 0.0f;

					m_Particles[boxIndex] = particle;
					m_Particles0[boxIndex] = DFSPHParticle0{ particle.Position, particle.Velocity };
					boxIndex++;
				}
			}
		}

		unsigned int threadStarts = 0;
		ComputeHelper::GetThreadBlocks(m_Info.ParticleCount, m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);

		m_VertexArray = Ref<VertexArray>::Create();
		m_VertexBuffer = Ref<VertexBuffer>::Create(m_Info.ParticleCount * sizeof(DFSPHParticle));
		m_VertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position"                         }, // Used
			{ ShaderDataType::Float3, "a_Velocity"                         }, // Used
			{ ShaderDataType::Float3, "a_Acceleration"                     }, // Used
			{ ShaderDataType::Float,  "a_Mass"                             }, 
			{ ShaderDataType::Float,  "a_Density"                          },
			{ ShaderDataType::Float,  "a_Kappa"                            },
			{ ShaderDataType::Float,  "a_KappaVelocity"                    },
			// Viscosity												   
			{ ShaderDataType::Float3, "a_ViscosityDifference"              },
			// Surface tension											   
			{ ShaderDataType::Float3, "a_MonteCarloSurfaceNormals"         },
			{ ShaderDataType::Float3, "a_MonteCarloSurfaceNormalsSmooth"   },
			{ ShaderDataType::Float,  "a_FinalCurvature"                   },
			{ ShaderDataType::Float,  "a_DeltaFinalCurvature"              },
			{ ShaderDataType::Float,  "a_SmoothedCurvature"                },
			{ ShaderDataType::Float,  "a_MonteCarloSurfaceCurvature"       },
			{ ShaderDataType::Float,  "a_MonteCarloSurfaceCurvatureSmooth" },
			{ ShaderDataType::Float,  "a_ClassifierInput"                  },
			{ ShaderDataType::Float,  "a_ClassifierOutput"                 }
		});

		m_VertexArray->AddVertexBuffer(m_VertexBuffer);
		m_VertexBuffer->SetData(0, m_Info.ParticleCount * sizeof(DFSPHParticle), m_Particles);
		m_VertexBuffer->Unbind();

		// Register buffer as a CUDA resource
		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_VertexBuffer->GetRendererID()))
	}

	void DFSPHImplementation::CalculateTimeStepSize(const thrust::device_ptr<DFSPHParticle>& mappedParticles)
	{
		m_MaxVelocityMagnitudeUnaryOperator.TimeStepSize = m_Info.TimeStepSize;
		CalculateMaxVelocityMagnitude(mappedParticles, 0.1f);

		// Use the highest velocity magnitude to approximate the new time step size
		m_Info.TimeStepSize = 0.4f * (m_Info.ParticleDiameter / sqrt(m_MaxVelocityMagnitude));
		m_Info.TimeStepSize = std::min(m_Info.TimeStepSize, m_Description.MaxTimeStepSize);
		m_Info.TimeStepSize = std::max(m_Info.TimeStepSize, m_Description.MinTimeStepSize);

		// Copy the memory new time step back to the device
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))
	}
	void DFSPHImplementation::CalculateMaxVelocityMagnitude(const thrust::device_ptr<DFSPHParticle>& mappedParticles, float initialValue)
	{
		m_MaxVelocityMagnitude = thrust::transform_reduce(
			mappedParticles,
			mappedParticles + m_Info.ParticleCount,
			m_MaxVelocityMagnitudeUnaryOperator,
			initialValue,
			thrust::maximum<float>()
		);
	}
}