#include "pch.h"
#include "DFSPHImplementation.h"

#include "Compute/ComputeHelper.h"

#include <cuda_gl_interop.h>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>

namespace vfd
{
	DFSPHImplementation::DFSPHImplementation(const DFSPHSimulationDescription& desc)
	{
		// Preallocate the necessary memory
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_PrecomputedSmoothingKernel), sizeof(PrecomputedDFSPHCubicKernel)))
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_Info), sizeof(DFSPHSimulationInfo)))

		SetDescription(desc);
	}

	DFSPHImplementation::~DFSPHImplementation()
	{
		delete[] m_Particles;

		COMPUTE_SAFE(cudaFree(d_Info))
		COMPUTE_SAFE(cudaFree(d_PrecomputedSmoothingKernel))
		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_VertexBuffer->GetRendererID()))
	}

	void DFSPHImplementation::Simulate()
	{
		std::cout << "**Simulation started\n";
	}

	void DFSPHImplementation::OnUpdate()
	{
		if (m_Info.ParticleCount == 0u) {
			return;
		}

		// Map OpenGL memory to CUDA memory
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))

		// Sort all particles based on their radius and position
		m_DebugInfo.NeighborhoodSearchTimer.Start();
		m_ParticleSearch.FindNeighbors(particles);
		d_NeighborSet = m_ParticleSearch.GetNeighborSet();
		m_DebugInfo.NeighborhoodSearchTimer.Stop();

		// Simulate
		{
			// Compute boundaries
			m_DebugInfo.BaseSolverTimer.Start();
			ComputeVolumeAndBoundaryKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info, 
				ComputeHelper::GetPointer(d_RigidBodies)
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute densities  
			ComputeDensityKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info,
				d_NeighborSet,
				ComputeHelper::GetPointer(d_RigidBodies),
				d_PrecomputedSmoothingKernel
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute factors 
			ComputeDFSPHFactorKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles, 
				d_Info, 
				d_NeighborSet,
				ComputeHelper::GetPointer(d_RigidBodies),
				d_PrecomputedSmoothingKernel
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())
			m_DebugInfo.BaseSolverTimer.Stop();

			m_DebugInfo.DivergenceSolverTimer.Start();
			ComputeDivergence(particles);
			m_DebugInfo.DivergenceSolverTimer.Stop();

			// Clear accelerations
			ClearAccelerationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles, 
				d_Info
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			m_DebugInfo.SurfaceTensionSolverTimer.Start();
			SolveSurfaceTension(particles);
			m_DebugInfo.SurfaceTensionSolverTimer.Stop();

			m_DebugInfo.ViscositySolverTimer.Start();
			ComputeViscosity(particles);
			m_DebugInfo.ViscositySolverTimer.Stop();

			// Update time step size
			ComputeTimeStepSize(thrust::device_pointer_cast(particles));
		
			// Calculate velocities
			ComputeVelocityKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			m_DebugInfo.PressureSolverTimer.Start();
		    ComputePressure(particles);
			m_DebugInfo.PressureSolverTimer.Stop();

			// Compute positions
			ComputePositionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())
		}

		// Unmap OpenGL memory 
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		m_DebugInfo.IterationCount++;
	}

	void DFSPHImplementation::Reset()
	{
		// Reset particle data
		m_VertexBuffer->SetData(0, m_Info.ParticleCount * sizeof(DFSPHParticle), m_Particles);
		m_VertexBuffer->Unbind();

		// Compute min max values of the current particle layout
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))
		ComputeMaxVelocityMagnitude(thrust::device_pointer_cast(particles), 0.0f);
		m_ParticleSearch.ComputeMinMax(particles);
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		m_DebugInfo.IterationCount = 0u;

		// Reset the time step size
		m_Info.TimeStepSize = m_Description.TimeStepSize;
		m_Info.TimeStepSize2 = m_Description.TimeStepSize * m_Description.TimeStepSize;
		m_Info.TimeStepSizeInverse = 1.0f / m_Info.TimeStepSize;
		m_Info.TimeStepSize2Inverse = 1.0f / m_Info.TimeStepSize2;

		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))
	}

	void DFSPHImplementation::SetFluidObjects(const std::vector<Ref<FluidObject>>& fluidObjects)
	{
		std::cout << "**Initializing fluid particles\n";
		std::cout << "Fluid object count: " << fluidObjects.size() << '\n';

		m_Info.ParticleCount = 0u;
		for(Ref<FluidObject> fluidObject : fluidObjects)
		{
			m_Info.ParticleCount += fluidObject->GetPositionCount();
		}

		std::cout << "Total particle count: " << m_Info.ParticleCount << '\n';

		// Free existing particles 
		delete[] m_Particles;

		m_Particles = new DFSPHParticle[m_Info.ParticleCount];
		unsigned int particleIndex = 0u;

		// Initialize particles 
		for (Ref<FluidObject> fluidObject : fluidObjects)
		{
			for(const glm::vec3& position : fluidObject->GetPositions())
			{
				DFSPHParticle particle{};

				// Particle data
				particle.Position = position;
				particle.Velocity = { 0.0f, 0.0f, 0.0f };
				particle.Acceleration = { 0.0f, 0.0f, 0.0f };
				particle.PressureAcceleration = { 0.0f, 0.0f, 0.0f };
				particle.PressureResiduum = 0.0f;
				particle.Density = 0.0f;
				particle.DensityAdvection = 0.0f;
				particle.PressureRho2 = 0.0f;
				particle.PressureRho2V = 0.0f;
				particle.Factor = 0.0f;

				// Viscosity
				particle.VelocityDifference = { 0.0f, 0.0f, 0.0f };

				// Surface tension
				particle.MonteCarloSurfaceNormal = { 0.0f, 0.0f, 0.0f };
				particle.MonteCarloSurfaceNormalSmooth = { 0.0f, 0.0f, 0.0f };
				particle.DeltaFinalCurvature = 0.0f;
				particle.MonteCarloSurfaceCurvature = 0.0f;
				particle.MonteCarloSurfaceCurvatureSmooth = 0.0f;

				m_Particles[particleIndex++] = particle;
			}
		}

		m_Residual = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_Preconditioner = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_PreconditionerZ = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_OperationTemporary = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_Temp = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_ViscosityGradientB = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_ViscosityGradientG = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_PreconditionerInverseDiagonal = thrust::device_vector<glm::mat3x3>(m_Info.ParticleCount);

		// Calculate block and thread counts
		unsigned int threadStarts = 0u;
		ComputeHelper::GetThreadBlocks(m_Info.ParticleCount, m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);

		// Initialize OpenGL buffers
		if(m_VertexBuffer)
		{
			// Free existing particles from the device 
			COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_VertexBuffer->GetRendererID()))
		}

		m_VertexArray = Ref<VertexArray>::Create();
		m_VertexBuffer = Ref<VertexBuffer>::Create(m_Info.ParticleCount * sizeof(DFSPHParticle));
		m_VertexBuffer->SetLayout({
			// Base solver
			{ ShaderDataType::Float3, "a_Position"                         },
			{ ShaderDataType::Float3, "a_Velocity"                         },
			{ ShaderDataType::Float3, "a_Acceleration"                     },
			{ ShaderDataType::Float3, "a_PressureAcceleration"             },
			{ ShaderDataType::Float,  "a_PressureResiduum"                 },
			{ ShaderDataType::Float,  "a_Density"                          },
			{ ShaderDataType::Float,  "a_DensityAdvection"                 },
			{ ShaderDataType::Float,  "a_PressureRho2"                     },
			{ ShaderDataType::Float,  "a_PressureRho2V"                    },
			{ ShaderDataType::Float,  "a_Factor"                           },
			// Viscosity												   
			{ ShaderDataType::Float3, "a_VelocityDifference"               },
			// Surface tension											 
			{ ShaderDataType::Float3, "a_MonteCarloSurfaceNormals"         },
			{ ShaderDataType::Float3, "a_MonteCarloSurfaceNormalsSmooth"   },
			{ ShaderDataType::Float,  "a_MonteCarloSurfaceCurvature"       },
			{ ShaderDataType::Float,  "a_MonteCarloSurfaceCurvatureSmooth" },
			{ ShaderDataType::Float,  "a_DeltaFinalCurvature"              }
		});

		m_VertexArray->AddVertexBuffer(m_VertexBuffer);
		m_VertexBuffer->SetData(0, m_Info.ParticleCount * sizeof(DFSPHParticle), m_Particles);
		m_VertexBuffer->Unbind();

		// Register buffer as a CUDA resource
		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_VertexBuffer->GetRendererID()))

		// Copy scene data over to the device
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))

		// Update the particle search particle count
		m_ParticleSearch.SetPointCount(m_Info.ParticleCount);

		// Compute min max values of the current particle layout
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))
		m_ParticleSearch.ComputeMinMax(particles);
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		std::cout << "Fluid objects initialized\n\n";
	}

	void DFSPHImplementation::SetRigidBodies(const std::vector<Ref<RigidBody>>& rigidBodies)
	{
		std::cout << "**Initializing rigid bodies\n";
		std::cout << "Rigid body count: " << rigidBodies.size() << '\n';

		m_RigidBodies = rigidBodies;
		m_Info.RigidBodyCount = static_cast<unsigned int>(rigidBodies.size());

		std::vector<RigidBodyDeviceData*> data;
		for (Ref<RigidBody> rb : rigidBodies)
		{
			data.push_back(rb->GetDeviceData());
		}
		d_RigidBodies = data;

		std::cout << "Rigid bodies initialized\n\n";
	}

	const Ref<VertexArray>& DFSPHImplementation::GetVertexArray() const
	{
		return m_VertexArray;
	}

	unsigned int DFSPHImplementation::GetParticleCount() const
	{
		return m_Info.ParticleCount;
	}

	float DFSPHImplementation::GetParticleRadius() const
	{
		return m_Info.ParticleRadius;
	}

	float DFSPHImplementation::GetMaxVelocityMagnitude() const
	{
		return m_MaxVelocityMagnitude;
	}

	float DFSPHImplementation::GetCurrentTimeStepSize() const
	{
		return m_Info.TimeStepSize;
	}

	const ParticleSearch& DFSPHImplementation::GetParticleSearch() const
	{
		return m_ParticleSearch;
	}

	const DFSPHSimulationDescription& DFSPHImplementation::GetDescription() const
	{
		return m_Description;
	}

	void DFSPHImplementation::SetDescription(const DFSPHSimulationDescription& desc)
	{
		m_Description = desc;

		if(m_Description.ParticleRadius != m_Info.ParticleRadius)
		{
			// Update the smoothing kernel radius
			m_PrecomputedSmoothingKernel.SetRadius(4.0f * m_Description.ParticleRadius);
			COMPUTE_SAFE(cudaMemcpy(d_PrecomputedSmoothingKernel, &m_PrecomputedSmoothingKernel, sizeof(PrecomputedDFSPHCubicKernel), cudaMemcpyHostToDevice))

			// Update the search radius 
			m_ParticleSearch.SetSearchRadius(4.0f * m_Description.ParticleRadius);
		}
	
		m_Info.ParticleRadius = m_Description.ParticleRadius;
		m_Info.ParticleDiameter = 2.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius = 4.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius2 = m_Info.SupportRadius * m_Info.SupportRadius;

		m_Info.Volume = 0.8f * m_Info.ParticleDiameter * m_Info.ParticleDiameter * m_Info.ParticleDiameter;
		m_Info.Density0 = 1000.0f;
		m_Info.ParticleMass = m_Info.Volume * m_Info.Density0;
		m_Info.ParticleMassInverse = 1.0f / m_Info.ParticleMass;

		// Viscosity
		m_Info.Viscosity = m_Description.Viscosity;
		m_Info.BoundaryViscosity = m_Description.BoundaryViscosity;
		m_Info.DynamicViscosity = m_Info.Viscosity * m_Info.Density0;
		m_Info.DynamicBoundaryViscosity = m_Info.BoundaryViscosity * m_Info.Density0;
		m_Info.TangentialDistanceFactor = m_Description.TangentialDistanceFactor;
		m_Info.TangentialDistance = m_Info.TangentialDistanceFactor * m_Info.SupportRadius;

		// Time step size
		m_Info.TimeStepSize = m_Description.TimeStepSize;
		m_Info.TimeStepSize2 = m_Description.TimeStepSize * m_Description.TimeStepSize;
		m_Info.TimeStepSizeInverse = 1.0f / m_Info.TimeStepSize;
		m_Info.TimeStepSize2Inverse = 1.0f / m_Info.TimeStepSize2;

		// Surface tension
		m_Info.SurfaceTension = m_Description.SurfaceTension;
		m_Info.ClassifierSlope = 74.688796680497925f;
		m_Info.ClassifierConstant = 12.0f;
		m_Info.TemporalSmoothing = m_Description.TemporalSmoothing;
		m_Info.SmoothingFactor = 0.5f;
		m_Info.Factor = 0.8f;
		m_Info.NeighborParticleRadius = m_Info.ParticleRadius * m_Info.Factor;

		m_Info.Gravity = m_Description.Gravity;
	}

	const DFSPHSimulationInfo& DFSPHImplementation::GetInfo() const
	{
		return m_Info;
	}

	PrecomputedDFSPHCubicKernel& DFSPHImplementation::GetKernel()
	{
		return m_PrecomputedSmoothingKernel;
	}

	const std::vector<Ref<RigidBody>>& DFSPHImplementation::GetRigidBodies() const
	{
		return m_RigidBodies;
	}

	unsigned int DFSPHImplementation::GetRigidBodyCount() const
	{
		return static_cast<unsigned int>(m_RigidBodies.size());
	}

	const DFSPHDebugInfo& DFSPHImplementation::GetDebugInfo() const
	{
		return m_DebugInfo;
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

		if(m_Description.CSDFix > 0)
		{
			m_Info.SurfaceTensionSampleCount = m_Description.CSDFix;
		}
		else
		{
			m_Info.SurfaceTensionSampleCount = static_cast<int>(static_cast<float>(m_Description.CSD) * m_Info.TimeStepSize);
		}

		m_Info.MonteCarloFactor = asin(m_Info.NeighborParticleRadius / m_Info.ParticleRadius);

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
		ComputeDensityAdvectionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info, 
			d_NeighborSet,
			ComputeHelper::GetPointer(d_RigidBodies),
			d_PrecomputedSmoothingKernel
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		const thrust::device_ptr<DFSPHParticle>& mappedParticles = thrust::device_pointer_cast(particles);
		const float eta = m_Description.MaxPressureSolverError * 0.0001f * m_Info.Density0;

		m_DensityErrorUnaryOperator.Density0 = m_Info.Density0;
		m_DebugInfo.PressureSolverIterationCount = 0u;
		m_DebugInfo.PressureSolverError = 0.0f;

		while ((m_DebugInfo.PressureSolverError > eta || m_DebugInfo.PressureSolverIterationCount < m_Description.MinPressureSolverIterations) && m_DebugInfo.PressureSolverIterationCount < m_Description.MaxPressureSolverIterations)
		{
			// Advance solver
			ComputePressureAccelerationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles, 
				d_Info,
				d_NeighborSet,
				ComputeHelper::GetPointer(d_RigidBodies),
				d_PrecomputedSmoothingKernel
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			PressureSolveIterationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info,
				d_NeighborSet, 
				ComputeHelper::GetPointer(d_RigidBodies),
				d_PrecomputedSmoothingKernel
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute solver error 
			const float densityError = thrust::transform_reduce(
				mappedParticles,
				mappedParticles + m_Info.ParticleCount,
				m_DensityErrorUnaryOperator,
				0.0f,
				thrust::minus<float>()
			);

			m_DebugInfo.PressureSolverError = densityError / static_cast<float>(m_Info.ParticleCount);
			m_DebugInfo.PressureSolverIterationCount++;
		}

		// Update particle velocities
		ComputePressureAccelerationAndVelocityKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info, 
			d_NeighborSet,
			ComputeHelper::GetPointer(d_RigidBodies),
			d_PrecomputedSmoothingKernel
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	void DFSPHImplementation::ComputeDivergence(DFSPHParticle* particles)
	{
		if(m_Description.EnableDivergenceSolverError == false)
		{

			m_DebugInfo.DivergenceSolverError = 0.0f;
			m_DebugInfo.DivergenceSolverIterationCount = 0u;
			return;
		}

		ComputeDensityChangeKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info, 
			d_NeighborSet,
			ComputeHelper::GetPointer(d_RigidBodies),
			d_PrecomputedSmoothingKernel
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		const thrust::device_ptr<DFSPHParticle>& mappedParticles = thrust::device_pointer_cast(particles);
		const float eta = m_Info.TimeStepSizeInverse * m_Description.MaxDivergenceSolverError * 0.0001f * m_Info.Density0;

		m_DensityErrorUnaryOperator.Density0 = m_Info.Density0;
		m_DebugInfo.DivergenceSolverIterationCount = 0u;
		m_DebugInfo.DivergenceSolverError = 0.0f;

		while ((m_DebugInfo.DivergenceSolverError > eta || m_DebugInfo.DivergenceSolverIterationCount < m_Description.MinDivergenceSolverIterations) && m_DebugInfo.DivergenceSolverIterationCount < m_Description.MaxDivergenceSolverIterations)
		{
			// Advance solver
			ComputePressureAccelerationAndDivergenceKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info, 
				d_NeighborSet,
				ComputeHelper::GetPointer(d_RigidBodies),
				d_PrecomputedSmoothingKernel
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			DivergenceSolveIterationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles, 
				d_Info, 
				d_NeighborSet, 
				ComputeHelper::GetPointer(d_RigidBodies),
				d_PrecomputedSmoothingKernel
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute solver error 
			const float densityError = thrust::transform_reduce(
				mappedParticles,
				mappedParticles + m_Info.ParticleCount,
				m_DensityErrorUnaryOperator,
				0.0f,
				thrust::minus<float>()
			);

			m_DebugInfo.DivergenceSolverError = densityError / static_cast<float>(m_Info.ParticleCount);
			m_DebugInfo.DivergenceSolverIterationCount++;
		}

		// Update particle velocities
		ComputePressureAccelerationAndFactorKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles, 
			d_Info, 
			d_NeighborSet,
			ComputeHelper::GetPointer(d_RigidBodies),
			d_PrecomputedSmoothingKernel
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	void DFSPHImplementation::ComputeViscosity(DFSPHParticle* particles)
	{
		if (m_Description.EnableViscositySolver == false)
		{
			m_DebugInfo.ViscositySolverError = 0.0f;
			m_DebugInfo.ViscositySolverIterationCount = 0u;
			return;
		}

		ComputeViscosityPreconditionerKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info, 
			d_NeighborSet,
			ComputeHelper::GetPointer(d_RigidBodies),
			d_PrecomputedSmoothingKernel, 
			ComputeHelper::GetPointer(m_PreconditionerInverseDiagonal)
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		ComputeViscosityGradientKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info,
			ComputeHelper::GetPointer(d_RigidBodies),
			d_PrecomputedSmoothingKernel,
			ComputeHelper::GetPointer(m_ViscosityGradientB),
			ComputeHelper::GetPointer(m_ViscosityGradientG)
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		// Solve with guess
		SolveViscosity(particles);

		ApplyViscosityForceKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info,
			ComputeHelper::GetPointer(m_ViscosityGradientG)
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	void DFSPHImplementation::SolveViscosity(DFSPHParticle* particles)
	{
		const auto residualNorm2ZipIterator = thrust::make_zip_iterator(thrust::make_tuple(m_Residual.begin(), m_Residual.begin()));
		const auto absNewZipIterator = thrust::make_zip_iterator(thrust::make_tuple(m_Residual.begin(), m_Preconditioner.begin()));
		const auto alphaZipIterator = thrust::make_zip_iterator(thrust::make_tuple(m_Preconditioner.begin(), m_Temp.begin()));
		const auto absNewPreconditionerZipIterator = thrust::make_zip_iterator(thrust::make_tuple(m_Residual.begin(), m_PreconditionerZ.begin()));

		m_DebugInfo.ViscositySolverIterationCount = 0u;
		const unsigned int n = m_Info.ParticleCount;

		ComputeMatrixVecProdFunctionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info,
			d_NeighborSet,
			ComputeHelper::GetPointer(d_RigidBodies),
			d_PrecomputedSmoothingKernel,
			ComputeHelper::GetPointer(m_ViscosityGradientG),
			ComputeHelper::GetPointer(m_OperationTemporary)
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		thrust::transform(
			m_ViscosityGradientB.begin(),
			m_ViscosityGradientB.end(),
			m_OperationTemporary.begin(),
			m_Residual.begin(),
			thrust::minus<glm::vec3>()
		);

		const float rhsNorm2 = thrust::transform_reduce(
			m_ViscosityGradientB.begin(),
			m_ViscosityGradientB.end(),
			m_SquaredNormUnaryOperator,
			0.0f,
			thrust::plus<float>()
		);

		if (rhsNorm2 == 0.0f)
		{
			thrust::fill(m_ViscosityGradientG.begin(), m_ViscosityGradientG.end(), glm::vec3(0.0f, 0.0f, 0.0f));
			m_DebugInfo.ViscositySolverError = 0.0f;
			return;
		}

		const float threshold = std::max((m_Description.MaxViscositySolverError * m_Description.MaxViscositySolverError) * 0.0001f * rhsNorm2, std::numeric_limits<float>::min());

		float residualNorm2 = thrust::transform_reduce(
			residualNorm2ZipIterator,
			residualNorm2ZipIterator + n,
			m_DotUnaryOperator,
			0.0f,
			thrust::plus<float>()
		);

		if (residualNorm2 < threshold)
		{
			m_DebugInfo.ViscositySolverError = sqrt(residualNorm2 / rhsNorm2);
			return;
		}

		thrust::transform(
			m_PreconditionerInverseDiagonal.begin(),
			m_PreconditionerInverseDiagonal.end(),
			m_Residual.begin(),
			m_Preconditioner.begin(),
			m_Vec3Mat3MultiplyBinaryOperator
		);

		float absNew = std::abs(thrust::transform_reduce(
			absNewZipIterator,
			absNewZipIterator + n,
			m_DotUnaryOperator,
			0.0f,
			thrust::plus<float>()
		));

		while (m_DebugInfo.ViscositySolverIterationCount >= m_Description.MinViscositySolverIterations && m_DebugInfo.ViscositySolverIterationCount < m_Description.MaxViscositySolverIterations)
		{
			ComputeMatrixVecProdFunctionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info,
				d_NeighborSet,
				ComputeHelper::GetPointer(d_RigidBodies),
				d_PrecomputedSmoothingKernel,
				ComputeHelper::GetPointer(m_Preconditioner),
				ComputeHelper::GetPointer(m_Temp)
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			thrust::transform(
				m_Preconditioner.begin(),
				m_Preconditioner.end(),
				m_Temp.begin(),
				m_OperationTemporary.begin(),
				thrust::multiplies<glm::vec3>()
			);

			const float alpha = absNew / thrust::transform_reduce(
				alphaZipIterator,
				alphaZipIterator + n,
				m_DotUnaryOperator,
				0.0f,
				thrust::plus<float>()
			);

			thrust::transform(
				m_Preconditioner.begin(),
				m_Preconditioner.end(),
				thrust::make_constant_iterator(alpha),
				m_OperationTemporary.begin(),
				m_Vec3FloatMultiplyBinaryOperator
			);

			thrust::transform(
				m_ViscosityGradientG.begin(),
				m_ViscosityGradientG.end(),
				m_OperationTemporary.begin(),
				m_ViscosityGradientG.begin(),
				thrust::plus<glm::vec3>()
			);

			thrust::transform(
				m_Temp.begin(),
				m_Temp.end(),
				thrust::make_constant_iterator(alpha),
				m_OperationTemporary.begin(),
				m_Vec3FloatMultiplyBinaryOperator
			);

			thrust::transform(
				m_Residual.begin(),
				m_Residual.end(),
				m_OperationTemporary.begin(),
				m_Residual.begin(),
				thrust::minus<glm::vec3>()
			);

			residualNorm2 = thrust::transform_reduce(
				m_Residual.begin(),
				m_Residual.end(),
				m_SquaredNormUnaryOperator,
				0.0f,
				thrust::plus<float>()
			);

			if (residualNorm2 < threshold) {
				break;
			}

			thrust::transform(
				m_PreconditionerInverseDiagonal.begin(),
				m_PreconditionerInverseDiagonal.end(),
				m_Residual.begin(),
				m_PreconditionerZ.begin(),
				m_Vec3Mat3MultiplyBinaryOperator
			);

			COMPUTE_SAFE(cudaDeviceSynchronize())

			const float absOld = absNew;

			absNew = std::abs(thrust::transform_reduce(
				absNewPreconditionerZipIterator,
				absNewPreconditionerZipIterator + n,
				m_DotUnaryOperator,
				0.0f,
				thrust::plus<float>()
			));

			const float beta = absNew / absOld;

			thrust::transform(
				m_Preconditioner.begin(),
				m_Preconditioner.end(),
				thrust::make_constant_iterator(beta),
				m_OperationTemporary.begin(),
				m_Vec3FloatMultiplyBinaryOperator
			);

			thrust::transform(
				m_OperationTemporary.begin(),
				m_OperationTemporary.end(),
				m_PreconditionerZ.begin(),
				m_Preconditioner.begin(),
				thrust::plus<glm::vec3>()
			);

			m_DebugInfo.ViscositySolverIterationCount++;
		}

		m_DebugInfo.ViscositySolverError = sqrt(residualNorm2 / rhsNorm2);
	}

	void DFSPHImplementation::SolveSurfaceTension(DFSPHParticle* particles)
	{
		if(m_Description.EnableSurfaceTensionSolver == false)
		{
			return;
		}

		ComputeSurfaceTensionClassificationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info,
			d_NeighborSet
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		ComputeSurfaceTensionNormalsAndCurvatureKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info,
			d_NeighborSet
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		for(unsigned int i = 0; i < m_Description.SurfaceTensionSmoothPassCount; i++)
		{
			ComputeSurfaceTensionBlendingKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info
			);
			COMPUTE_SAFE(cudaDeviceSynchronize())
		}
	}
}