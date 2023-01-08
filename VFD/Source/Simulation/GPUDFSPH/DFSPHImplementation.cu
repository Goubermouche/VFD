﻿#include "pch.h"
#include "DFSPHImplementation.h"

#include <cuda_gl_interop.h>
#include "Compute/ComputeHelper.h"
#include "Core/Math/Math.h"

#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include <functional>

namespace vfd
{
	DFSPHImplementation::DFSPHImplementation(const GPUDFSPHSimulationDescription& desc)
		 : m_Description(desc)
	{
		InitFluidData();

		// Init smoothing kernels
		m_PrecomputedSmoothingKernel.SetRadius(m_Info.SupportRadius);
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_PrecomputedSmoothingKernel), sizeof(PrecomputedDFSPHCubicKernel)))
		COMPUTE_SAFE(cudaMemcpy(d_PrecomputedSmoothingKernel, &m_PrecomputedSmoothingKernel, sizeof(PrecomputedDFSPHCubicKernel), cudaMemcpyHostToDevice))

		InitRigidBodies();

		// Copy scene data over to the device
		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_Info), sizeof(DFSPHSimulationInfo)))
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))
		
		// Neighborhood search
		m_ParticleSearch = new ParticleSearch(m_Info.ParticleCount, m_Info.SupportRadius);

		// Compute min max values of the current particle layout
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))
		m_ParticleSearch->ComputeMinMax(particles);
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))
	}

	DFSPHImplementation::~DFSPHImplementation()
	{
		delete m_ParticleSearch;
		delete[] m_Particles;

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

			ComputeDivergence(particles);

			// Clear accelerations
			ClearAccelerationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			ComputeViscosity(particles);

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
		// Reset particle data
		m_VertexBuffer->SetData(0, m_Info.ParticleCount * sizeof(DFSPHParticle), m_Particles);
		m_VertexBuffer->Unbind();

		// Compute min max values of the current particle layout
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))
		ComputeMaxVelocityMagnitude(thrust::device_pointer_cast(particles), 0.0f);
		m_ParticleSearch->ComputeMinMax(particles);
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		m_IterationCount = 0;

		// Reset the time step size
		m_Info.TimeStepSize = m_Description.TimeStepSize;
		m_Info.TimeStepSize2 = m_Description.TimeStepSize * m_Description.TimeStepSize;
		m_Info.TimeStepSizeInverse = 1.0f / m_Info.TimeStepSize;
		m_Info.TimeStepSize2Inverse = 1.0f / m_Info.TimeStepSize2;

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

	float DFSPHImplementation::GetParticleRadius() const
	{
		return m_Info.ParticleRadius;
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

	const GPUDFSPHSimulationDescription& DFSPHImplementation::GetDescription() const
	{
		return m_Description;
	}

	void DFSPHImplementation::SetDescription(const GPUDFSPHSimulationDescription& desc)
	{
		m_Description = desc;

		m_Info.ParticleRadius = m_Description.ParticleRadius;
		m_Info.ParticleDiameter = 2.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius = 4.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius2 = m_Info.SupportRadius * m_Info.SupportRadius;

		m_Info.Volume = 0.8f * m_Info.ParticleDiameter * m_Info.ParticleDiameter * m_Info.ParticleDiameter;
		m_Info.Density0 = 1000.0f;
		m_Info.Gravity = m_Description.Gravity;

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
	}

	void DFSPHImplementation::InitRigidBodies()
	{
		//for (const RigidBodyDescription& desc : m_Description.BoundaryObjects)
		//{

		//}

		m_Info.RigidBodyCount = static_cast<unsigned>(m_Description.BoundaryObjects.size());
		//d_RigidBodyData = rigidBodies[0]->GetDeviceData(m_Info.ParticleCount);

		const RigidBodyDescription& desc = m_Description.BoundaryObjects[0];
		m_RigidBodies.push_back(Ref<RigidBody>::Create(desc, m_Info, m_PrecomputedSmoothingKernel));
		d_RigidBodyData = m_RigidBodies[0]->GetDeviceData();
	}

	void DFSPHImplementation::InitFluidData()
	{
		const glm::vec3 boxPosition = { 0.0f, 6.0f, 0.0f };
		const glm::uvec3 boxSize = { 40, 40, 40 };

		const glm::vec3 boxHalfSize = static_cast<glm::vec3>(boxSize - glm::uvec3(1)) / 2.0f;
		unsigned int boxIndex = 0u;

		m_Info.ParticleCount = glm::compMul(boxSize);
		m_Info.ParticleRadius = m_Description.ParticleRadius;
		m_Info.ParticleDiameter = 2.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius = 4.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius2 = m_Info.SupportRadius * m_Info.SupportRadius;
		
		m_Info.Volume = 0.8f * m_Info.ParticleDiameter * m_Info.ParticleDiameter * m_Info.ParticleDiameter;
		m_Info.Density0 = 1000.0f;
		m_Info.Gravity = m_Description.Gravity;

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

		m_Particles = new DFSPHParticle[m_Info.ParticleCount];

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
					particle.Density = 0.0f;
					particle.DensityAdvection = 0.0f;
					particle.PressureRho2 = 0.0f;
					particle.PressureRho2V = 0.0f;
					particle.Factor = 0.0f;

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

					m_Particles[boxIndex++] = particle;
				}
			}
		}

		m_Residual                      = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_Preconditioner                = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_PreconditionerZ               = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_OperationTemporary            = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_Temp                          = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_ViscosityGradientB            = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_ViscosityGradientG            = thrust::device_vector<glm::vec3>(m_Info.ParticleCount);
		m_PreconditionerInverseDiagonal = thrust::device_vector<glm::mat3x3>(m_Info.ParticleCount);

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
		const float eta = m_Description.MaxPressureSolverError * 0.0001f * m_Info.Density0;

		m_DensityErrorUnaryOperator.Density0 = m_Info.Density0;
		m_PressureSolverIterationCount = 0u;
		m_PressureSolverError = 0.0f;

		while ((m_PressureSolverError > eta || m_PressureSolverIterationCount < m_Description.MinPressureSolverIterations) && m_PressureSolverIterationCount < m_Description.MaxPressureSolverIterations)
		{
			// Advance solver
			ComputePressureAccelerationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			PressureSolveIterationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute solver error 
			const float densityError = thrust::transform_reduce(
				mappedParticles,
				mappedParticles + m_Info.ParticleCount,
				m_DensityErrorUnaryOperator,
				0.0f,
				thrust::minus<float>()
			);

			m_PressureSolverError = densityError / static_cast<float>(m_Info.ParticleCount);
			m_PressureSolverIterationCount++;
		}

		// Update particle velocities
		ComputePressureAccelerationAndVelocityKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	void DFSPHImplementation::ComputeDivergence(DFSPHParticle* particles)
	{
		if(m_Description.EnableDivergenceSolverError == false)
		{
			return;
		}

		ComputeDensityChangeKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		const thrust::device_ptr<DFSPHParticle>& mappedParticles = thrust::device_pointer_cast(particles);
		const float eta = m_Info.TimeStepSizeInverse * m_Description.MaxDivergenceSolverError * 0.0001f * m_Info.Density0;

		m_DensityErrorUnaryOperator.Density0 = m_Info.Density0;
		m_DivergenceSolverIterationCount = 0u;
		m_DivergenceSolverError = 0.0f;

		while ((m_DivergenceSolverError > eta || m_DivergenceSolverIterationCount < m_Description.MinDivergenceSolverIterations) && m_DivergenceSolverIterationCount < m_Description.MaxDivergenceSolverIterations)
		{
			// Advance solver
			ComputePressureAccelerationAndDivergenceKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			DivergenceSolveIterationKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Compute solver error 
			const float densityError = thrust::transform_reduce(
				mappedParticles,
				mappedParticles + m_Info.ParticleCount,
				m_DensityErrorUnaryOperator,
				0.0f,
				thrust::minus<float>()
			);

			m_DivergenceSolverError = densityError / static_cast<float>(m_Info.ParticleCount);
			m_DivergenceSolverIterationCount++;
		}

		// Update particle velocities
		ComputePressureAccelerationAndFactorKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info, d_NeighborSet, d_RigidBodyData, d_PrecomputedSmoothingKernel);
		COMPUTE_SAFE(cudaDeviceSynchronize())
	}

	void DFSPHImplementation::ComputeViscosity(DFSPHParticle* particles)
	{
		ComputeViscosityPreconditionerKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info, 
			d_NeighborSet,
			d_RigidBodyData, 
			d_PrecomputedSmoothingKernel, 
			ComputeHelper::GetPointer(m_PreconditionerInverseDiagonal)
		);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		ComputeViscosityGradientKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info,
			d_RigidBodyData,
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

		m_ViscositySolverIterationCount = 0u;
		const unsigned int n = m_Info.ParticleCount;

		ComputeMatrixVecProdFunctionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
			particles,
			d_Info,
			d_NeighborSet,
			d_RigidBodyData,
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
			m_ViscositySolverError = 0.0f;
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
			m_ViscositySolverError = sqrt(residualNorm2 / rhsNorm2);
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

		while (m_ViscositySolverIterationCount >= m_Description.MinViscositySolverIterations && m_ViscositySolverIterationCount < m_Description.MaxViscositySolverIterations)
		{
			ComputeMatrixVecProdFunctionKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (
				particles,
				d_Info,
				d_NeighborSet,
				d_RigidBodyData,
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

			m_ViscositySolverIterationCount++;
		}

		m_ViscositySolverError = sqrt(residualNorm2 / rhsNorm2);
	}
}