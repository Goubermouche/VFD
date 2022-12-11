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
	DFSPHImplementation::DFSPHImplementation()
	{

		//{
		//	const int N = 1000000;
		//	float* h_vec = (float*)malloc(N * sizeof(float));
		//	for (int i = 0; i < N; i++) {
		//		h_vec[i] = i;
		//	}

		//	float* d_vec;
		//	COMPUTE_SAFE(cudaMalloc((void**)&d_vec, N * sizeof(float)));
		//	COMPUTE_SAFE(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));

		//	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_vec);

		//	thrust::device_ptr<float> min_ptr;
		//	{
		//		TIME_SCOPE("max")
		//		min_ptr = thrust::max_element(dev_ptr, dev_ptr + N);
		//	}

		//	float min_value = min_ptr[0];
		//	printf("\nMininum value = %f\n", min_value);
		//	printf("Position = %i\n", &min_ptr[0] - &dev_ptr[0]);
		//}

		//{
		//	const int N = 1000;
		//	float* h_vec = (float*)malloc(N * sizeof(float));
		//	for (int i = 0; i < N; i++) {
		//		h_vec[i] = i;
		//	}

		//	float* d_vec;
		//	COMPUTE_SAFE(cudaMalloc((void**)&d_vec, N * sizeof(float)));
		//	COMPUTE_SAFE(cudaMemcpy(d_vec, h_vec, N * sizeof(float), cudaMemcpyHostToDevice));

		//	square        unary_op;
		//	thrust::maximum<float> binary_op;
		//	float init = 0;

		//	thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_vec);

		//	float result = thrust::transform_reduce(dev_ptr, dev_ptr + N, unary_op, init, binary_op);

		//	printf("\nMininum value = %f\n", result);
		//}

		InitFluidData();

		// Neighborhood search
		m_NeighborhoodSearch = new NeighborhoodSearch(m_Info.SupportRadius);
		m_NeighborhoodSearch->AddPointSet(m_Particles, m_Info.ParticleCount, true, true, true);
	}

	DFSPHImplementation::~DFSPHImplementation()
	{
		delete m_NeighborhoodSearch;
		delete[] m_Particles;

		COMPUTE_SAFE(cudaFree(d_Info))
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
		m_NeighborhoodSearch->FindNeighbors();
		if (m_IterationCount % 500 == 0) {
			PointSet& pointSet = m_NeighborhoodSearch->GetPointSet(0);
			pointSet.SortField(particles);
		}

		// Simulate
		{
			// Clear accelerations
			ClearAccelerationsKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Update time step size
			CalculateTimeStepSize(thrust::device_pointer_cast(particles));
		
			// Calculate velocities
			CalculateVelocitiesKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >>> (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())

			// Calculate positions
			CalculatePositionsKernel <<< m_BlockStartsForParticles, m_ThreadsPerBlock >> > (particles, d_Info);
			COMPUTE_SAFE(cudaDeviceSynchronize())
		}

		// Unmap OpenGL memory 
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		// Debug, after the offline solution gets properly implemented this function only needs to be called once
		// after the simulation finishes baking.
		// COMPUTE_SAFE(cudaMemcpy(&m_Info, d_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyDeviceToHost))

		m_IterationCount++;
	}

	const Ref<VertexArray>& DFSPHImplementation::GetVertexArray() const
	{
		return m_VertexArray;
	}

	void DFSPHImplementation::InitFluidData()
	{
		glm::ivec3 boxSize = { 20, 20, 20 };
		glm::vec3 boxHalfSize = static_cast<glm::vec3>(boxSize - glm::ivec3(1)) / 2.0f;
		unsigned int boxIndex = 0;

		m_Info.ParticleCount = glm::compMul(boxSize);
		m_Info.ParticleRadius = 0.025f;
		m_Info.ParticleDiameter = 2.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius = 4.0f * m_Info.ParticleRadius;
		m_Info.TimeStepSize = 0.001f;
		m_Info.Volume = 0.8f * m_Info.ParticleDiameter * m_Info.ParticleDiameter * m_Info.ParticleDiameter;
		m_Info.Density0 = 1000.0f;
		m_Info.WZero = 0.0f;
		m_Info.Gravity = { 0.0f, -9.81f, 0.0f };

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_Info), sizeof(DFSPHSimulationInfo)))
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))

		m_Particles = new DFSPHParticle[m_Info.ParticleCount];

		// Generate a simple box for the purposes of testing 
		for (int x = 0; x < boxSize.x; x++)
		{
			for (int y = 0; y < boxSize.y; y++)
			{
				for (int z = 0; z < boxSize.z; z++)
				{
					DFSPHParticle particle{};

					// Particle data
					particle.Position = {
						(static_cast<float>(x) - boxHalfSize.x) * m_Info.ParticleDiameter,
						(static_cast<float>(y) - boxHalfSize.y) * m_Info.ParticleDiameter,
						(static_cast<float>(z) - boxHalfSize.z) * m_Info.ParticleDiameter
					};

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

					m_Particles[boxIndex++] = particle;
				}
			}
		}

		unsigned int threadStarts = 0;
		ComputeHelper::GetThreadBlocks(m_Info.ParticleCount, m_ThreadsPerBlock, m_BlockStartsForParticles, threadStarts);

		m_VertexArray = Ref<VertexArray>::Create();
		m_VertexBuffer = Ref<VertexBuffer>::Create(m_Info.ParticleCount * sizeof(DFSPHParticle));
		m_VertexBuffer->SetLayout({
			{ ShaderDataType::Float3, "a_Position"                         },
			{ ShaderDataType::Float3, "a_Velocity"                         },
			{ ShaderDataType::Float3, "a_Acceleration"                     },
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

		// Find the highest velocity magnitude in the simulation
		const float maxVelocityMagnitude = thrust::transform_reduce(
			mappedParticles, 
			mappedParticles + m_Info.ParticleCount,
			m_MaxVelocityMagnitudeUnaryOperator,
			0.1f, 
			thrust::maximum<float>());

		// Use the highest velocity magnitude to approximate the new time step size
		m_Info.TimeStepSize = 0.4f * (m_Info.ParticleDiameter / sqrt(maxVelocityMagnitude));
		m_Info.TimeStepSize = std::min(m_Info.TimeStepSize, m_CFLMaxTimeStepSize);
		m_Info.TimeStepSize = std::max(m_Info.TimeStepSize, m_CFLMinTimeStepSize);

		// Copy the memory new time step back to the device
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))
	}
}