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
	//__global__ void TestKernel(RigidBodyDeviceData* rigidBody)
	//{
	//	glm::vec3 xi(0, 0, 0);
	//	glm::vec3 t(0, -0.25, 0); /////////
	//	const glm::mat3& rotationMatrix = rigidBody->Rotation;
	//	const glm::dvec3 localXi = glm::transpose(rotationMatrix) * (xi - t);

	//	glm::dvec3 c0;
	//	unsigned int cell[32];
	//	double N[32];
	//	glm::dvec3 dN[32];
	//	glm::dvec3 normal;

	//	rigidBody->Map->DetermineShapeFunction(0, localXi, cell, c0, N, dN);
	//	double dist = rigidBody->Map->Interpolate(0, cell, c0, N, normal, dN);
	//	const double volume = rigidBody->Map->Interpolate(1, cell, c0, N);

	//	//printf("   cell:\n");
	//	//for (int i = 0; i < 32; i++)
	//	//{
	//	//	printf("%u\n", cell[i]);
	//	//}
	//	//printf("\n   c0:   %f %f %f\n\n", c0.x, c0.y, c0.z);
	//	//printf("   N:\n");
	//	//for (int i = 0; i < 32; i++)
	//	//{
	//	//	printf("%f\n", N[i]);
	//	//}
	//	//printf("\n   volume: %.17g\n", volume);
	//}

	DFSPHImplementation::DFSPHImplementation(const GPUDFSPHSimulationDescription& desc, std::vector<Ref<RigidBody>>& rigidBodies)
		 : m_Description(desc)
	{
		InitFluidData();
		InitRigidBodies(rigidBodies);

		// Init smoothing kernels
		m_PrecomputedSmoothingKernel.SetRadius(m_Info.SupportRadius);
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_PrecomputedSmoothingKernel), sizeof(PrecomputedDFSPHCubicKernel)))
		COMPUTE_SAFE(cudaMemcpy(d_PrecomputedSmoothingKernel, &m_PrecomputedSmoothingKernel, sizeof(PrecomputedDFSPHCubicKernel), cudaMemcpyHostToDevice))

		// Copy scene data over to the device
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_Info), sizeof(DFSPHSimulationInfo)))
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))
		
		// Neighborhood search
		m_ParticleSearch = new ParticleSearch(m_Info.ParticleCount, m_Info.SupportRadius);

		// TestKernel << < 1, 1 >> > (d_RigidBodyData);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	DFSPHImplementation::~DFSPHImplementation()
	{
		// delete m_NeighborhoodSearch;
		delete m_ParticleSearch;
		delete[] m_Particles;
		delete[] m_Particles0;

		COMPUTE_SAFE(cudaFree(d_Info))
		COMPUTE_SAFE(cudaFree(d_PrecomputedSmoothingKernel))
		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_VertexBuffer->GetRendererID()))
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
		m_ParticleSearch->FindNeighbors(particles);
		if (m_IterationCount % 500 == 0) {
			m_ParticleSearch->Sort(particles);
		}

		d_NeighborSet = m_ParticleSearch->GetNeighborSet();

		// Simulate
		{
			// Compute boundaries
			ComputeVolumeAndBoundaryKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_RigidBodyData);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute densities 
			ComputeDensityKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute factors 
			ComputeDFSPHFactorKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// ComputeDivergence(particles);

			// Clear accelerations
			ClearAccelerationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Update time step size
			ComputeTimeStepSize(thrust::device_pointer_cast(particles));
		
			// Calculate velocities
			ComputeVelocityKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			ComputePressure(particles);

			// Compute positions
			ComputePositionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())
		}

		// Unmap OpenGL memory 
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

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
			particle.PressureAcceleration = { 0.0f, 0.0f, 0.0f };
			particle.PressureResiduum = 0.0f;
			particle.Mass = m_Info.Volume * m_Info.Density0;
			particle.Density = m_Info.Density0;
			particle.DensityAdvection = 0.0f;
			particle.PressureRho2 = 0.0f;
			particle.PressureRho2V = 0.0f;
			particle.Factor = 0.0f;
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

		ComputeMaxVelocityMagnitude(thrust::device_pointer_cast(particles), 0.0f);
		m_ParticleSearch->FindNeighbors(particles);

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

	const ParticleSearch* DFSPHImplementation::GetParticleSearch() const
	{
		return m_ParticleSearch;
	}

	void DFSPHImplementation::InitRigidBodies(std::vector<Ref<RigidBody>>& rigidBodies)
	{
		m_Info.RigidBodyCount = static_cast<unsigned>(rigidBodies.size());
		d_RigidBodyData = rigidBodies[0]->GetDeviceData(m_Info.ParticleCount);
	}

	void DFSPHImplementation::InitFluidData()
	{
		const glm::vec3 boxPosition = { 0.0f, 2.0f, 0.0f };
		const glm::uvec3 boxSize = { 20, 20, 20 };

		const glm::vec3 boxHalfSize = static_cast<glm::vec3>(boxSize - glm::uvec3(1)) / 2.0f;
		unsigned int boxIndex = 0;

		m_Info.ParticleCount = glm::compMul(boxSize);
		m_Info.ParticleRadius = m_Description.ParticleRadius;
		m_Info.ParticleDiameter = 2.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius = 4.0f * m_Info.ParticleRadius;
		m_Info.TimeStepSize = m_Description.TimeStepSize;
		m_Info.TimeStepSize2 = m_Description.TimeStepSize * m_Description.TimeStepSize;
		m_Info.TimeStepSizeInverse = 1.0f / m_Info.TimeStepSize;
		m_Info.TimeStepSize2Inverse = 1.0f / m_Info.TimeStepSize2;
		m_Info.Volume = 0.8f * m_Info.ParticleDiameter * m_Info.ParticleDiameter * m_Info.ParticleDiameter;
		m_Info.Density0 = 1000.0f;
		m_Info.Gravity = m_Description.Gravity;

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
					particle.PressureAcceleration = { 0.0f, 0.0f, 0.0f };
					particle.PressureResiduum = 0.0f;
					particle.Mass = m_Info.Volume * m_Info.Density0;
					particle.Density = m_Info.Density0;
					particle.DensityAdvection = 0.0f;
					particle.PressureRho2 = 0.0f;
					particle.PressureRho2V = 0.0f;
					particle.Factor = 0.0f;
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
			{ ShaderDataType::Float3, "a_PressureAcceleration"             }, // Used
			{ ShaderDataType::Float,  "a_PressureResiduum"                 }, // Used
			{ ShaderDataType::Float,  "a_Mass"                             }, // Used, TODO: check if necessary 
			{ ShaderDataType::Float,  "a_Density"                          }, // Used
			{ ShaderDataType::Float,  "a_DensityAdvection"                 }, // Used
			{ ShaderDataType::Float,  "a_PressureRho2"                     }, // Used
			{ ShaderDataType::Float,  "a_PressureRho2V"                    }, // Used
			{ ShaderDataType::Float,  "a_Factor"                           }, // Used
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

	void DFSPHImplementation::ComputeTimeStepSize(const thrust::device_ptr<DFSPHParticle>& mappedParticles)
	{
		ComputeMaxVelocityMagnitude(mappedParticles, 0.1f);

		// Prevent division by 0
		if (m_MaxVelocityMagnitude < 1.0e-9f) {
			m_MaxVelocityMagnitude = 1.0e-9f;
		}

		// Use the highest velocity magnitude to approximate the new time step size
		m_Info.TimeStepSize = 0.4f * (m_Info.ParticleDiameter / sqrt(m_MaxVelocityMagnitude));
		m_Info.TimeStepSize = std::min(m_Info.TimeStepSize, m_Description.MaxTimeStepSize);
		m_Info.TimeStepSize = std::max(m_Info.TimeStepSize, m_Description.MinTimeStepSize);
		m_Info.TimeStepSize2 = m_Info.TimeStepSize * m_Info.TimeStepSize;
		m_Info.TimeStepSizeInverse = 1.0f / m_Info.TimeStepSize;
		m_Info.TimeStepSize2Inverse = 1.0f / m_Info.TimeStepSize2;

		// Copy the memory new time step back to the device
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))
	}

	void DFSPHImplementation::ComputeMaxVelocityMagnitude(const thrust::device_ptr<DFSPHParticle>& mappedParticles, float initialValue)
	{
		m_MaxVelocityMagnitudeUnaryOperator.TimeStepSize = m_Info.TimeStepSize;

		m_MaxVelocityMagnitude = thrust::transform_reduce(
			mappedParticles,
			mappedParticles + m_Info.ParticleCount,
			m_MaxVelocityMagnitudeUnaryOperator,
			initialValue,
			thrust::maximum<float>()
		);
	}

	void DFSPHImplementation::ComputePressure(DFSPHParticle* particles)
	{
		ComputeDensityAdvectionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		const thrust::device_ptr<DFSPHParticle>& mappedParticles = thrust::device_pointer_cast(particles);
		bool chk = false;
		unsigned int pressureSolverIterations = 0;

		m_DensityErrorUnaryOperator.Density0 = m_Info.Density0;

		while ((chk == false || pressureSolverIterations < m_Description.MinPressureSolverIterations) && pressureSolverIterations < m_Description.MaxPressureSolverIterations)
		{
			ComputePressureAccelerationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			PressureSolveIterationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			const float densityError = thrust::transform_reduce(
				mappedParticles,
				mappedParticles + m_Info.ParticleCount,
				m_DensityErrorUnaryOperator,
				0.0f,
				thrust::minus<float>()
			);

			const float averageDensityError = densityError / static_cast<float>(m_Info.ParticleCount);
			const float eta = m_Description.MaxPressureSolverError * 0.01f * m_Info.Density0;
			chk = averageDensityError <= eta;

			pressureSolverIterations++;
		}

		ComputePressureAccelerationAndVelocityKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	void DFSPHImplementation::ComputeDivergence(DFSPHParticle* particles)
	{
		ComputeDensityChangeKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		const thrust::device_ptr<DFSPHParticle>& mappedParticles = thrust::device_pointer_cast(particles);
		bool chk = false;
		unsigned int divergenceSolverIterations = 0;

		constexpr unsigned int maxDivergenceSolverIterations = 100;
		constexpr float maxError = 0.001f;

		m_DensityErrorUnaryOperator.Density0 = m_Info.Density0;

		while((chk == false) && divergenceSolverIterations < maxDivergenceSolverIterations)
		{
			ComputePressureAccelerationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			const float densityError = thrust::transform_reduce(
				mappedParticles,
				mappedParticles + m_Info.ParticleCount,
				m_DensityErrorUnaryOperator,
				0.0f,
				thrust::minus<float>()
			);

			const float averageDensityError = densityError / static_cast<float>(m_Info.ParticleCount);
			const float eta = (m_Info.TimeStepSizeInverse) * maxError * 0.01f * m_Info.Density0;
			chk = averageDensityError <= eta;
			
			divergenceSolverIterations++;
		}

		ComputePressureAccelerationAndFactorKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}
}