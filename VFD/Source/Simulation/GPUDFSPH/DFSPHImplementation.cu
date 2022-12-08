#include "pch.h"
#include "DFSPHImplementation.h"

#include <cuda_gl_interop.h>
#include "Compute/ComputeHelper.h"
#include "DFSPHKernels.cuh"

namespace vfd
{
	DFSPHImplementation::DFSPHImplementation()
	{
		InitFluidData();

		// Neighborhood search
		m_NeighborhoodSearch = new NeighborhoodSearch(m_Info.SupportRadius);
		m_NeighborhoodSearch->AddPointSet(m_Particles, m_Info.ParticleCount, true, true, true);
		m_NeighborhoodSearch->FindNeighbors();
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
		if (m_IterationCount % 500 == 0) {
			PointSet& pointSet = m_NeighborhoodSearch->GetPointSet(0);
			pointSet.SortField(particles);
		}

		m_NeighborhoodSearch->FindNeighbors();

		// Run a basic test kernel
		//TestKernel<<< 1, 3 >>>(particles, d_Info);
		//COMPUTE_SAFE(cudaDeviceSynchronize())

		// Unmap OpenGL memory 
		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		// Debug, after the offline solution gets properly implemented this function only needs to be called once
		// after the simulation finishes baking.
		COMPUTE_SAFE(cudaMemcpy(&m_Info, d_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyDeviceToHost))

		m_IterationCount++;
	}

	const Ref<VertexArray>& DFSPHImplementation::GetVertexArray() const
	{
		return m_VertexArray;
	}

	void DFSPHImplementation::InitFluidData()
	{
		glm::ivec3 cubeSize = { 10, 10, 10 };
		glm::vec3 halfCubeSize = static_cast<glm::vec3>(cubeSize - glm::ivec3(1)) / 2.0f;
		unsigned int cubeIndex = 0;

		m_Info.ParticleCount = glm::compMul(cubeSize);
		m_Info.ParticleRadius = 0.025f;
		m_Info.ParticleDiameter = 2.0f * m_Info.ParticleRadius;
		m_Info.SupportRadius = 4.0f * m_Info.ParticleRadius;
		m_Info.TimeStepSize = 0.0f;
		m_Info.Volume = 0.8f * m_Info.ParticleDiameter * m_Info.ParticleDiameter * m_Info.ParticleDiameter;
		m_Info.Density0 = 1000.0f;
		m_Info.WZero = 0.0f;
		m_Info.Gravity = { 0.0f, -9.81f, 0.0f };

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_Info), sizeof(DFSPHSimulationInfo)))
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))

		m_Particles = new DFSPHParticle[m_Info.ParticleCount];

		for (int x = 0; x < cubeSize.x; x++)
		{
			for (int y = 0; y < cubeSize.y; y++)
			{
				for (int z = 0; z < cubeSize.z; z++)
				{
					DFSPHParticle particle{};

					// Particle data
					particle.Position = {
						(static_cast<float>(x) - halfCubeSize.x) * m_Info.ParticleDiameter,
						(static_cast<float>(y) - halfCubeSize.y) * m_Info.ParticleDiameter,
						(static_cast<float>(z) - halfCubeSize.z) * m_Info.ParticleDiameter
					};

					particle.Velocity = { 0.8f, 0.0f, 0.8f };
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

					m_Particles[cubeIndex++] = particle;
				}
			}
		}

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
}