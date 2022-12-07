﻿#include "pch.h"
#include "DFSPHImplementation.h"

#include <cuda_gl_interop.h>
#include "Compute/ComputeHelper.h"
#include "DFSPHKernels.cuh"

namespace vfd
{
	DFSPHImplementation::DFSPHImplementation()
	{
		m_Info.ParticleCount = 3;
		m_Info.SupportRadius = 0.0f;
		m_Info.TimeStepSize = 0.0f;
		m_Info.Volume = 0.0f;
		m_Info.Density0 = 0.0f;
		m_Info.WZero = 0.0f;

		COMPUTE_SAFE(cudaMalloc(reinterpret_cast<void**>(&d_Info), sizeof(DFSPHSimulationInfo)))
		COMPUTE_SAFE(cudaMemcpy(d_Info, &m_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyHostToDevice))

		m_Particles = new DFSPHParticle[3];

		for (size_t i = 0; i < 3; i++)
		{
			// Particle data
			DFSPHParticle particle{};
			particle.Position = { i, i, i };
			particle.Velocity = { 0.8f, 0.0f, 0.8f };
			particle.Acceleration = { 0.0f, 0.0f, 0.0f };

			particle.Mass = 0.0f;
			particle.Density = 0.0f;
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

		m_VertexArray = Ref<VertexArray>::Create();
		m_VertexBuffer = Ref<VertexBuffer>::Create(3 * sizeof(DFSPHParticle));
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
		m_VertexBuffer->SetData(0, 3 * sizeof(DFSPHParticle), m_Particles);
		m_VertexBuffer->Unbind();

		COMPUTE_SAFE(cudaGLRegisterBufferObject(m_VertexBuffer->GetRendererID()))

		// Neighborhood search
		m_NeighborhoodSearch = new NeighborhoodSearch(0.1f);
		m_NeighborhoodSearch->AddPointSet(m_Particles, 3, true, true, true);
	}

	DFSPHImplementation::~DFSPHImplementation()
	{
		delete[] m_Particles;
		delete m_NeighborhoodSearch;

		COMPUTE_SAFE(cudaFree(d_Info))
		COMPUTE_SAFE(cudaGLUnregisterBufferObject(m_VertexBuffer->GetRendererID()))
	}

	void DFSPHImplementation::OnUpdate()
	{
		DFSPHParticle* particles;
		COMPUTE_SAFE(cudaGLMapBufferObject(reinterpret_cast<void**>(&particles), m_VertexBuffer->GetRendererID()))

		if (m_IterationCount % 500 == 0) {
			// if (m_ParticleCount > 0) {
			const PointSet& pointSet = m_NeighborhoodSearch->GetPointSet(0);
			pointSet.SortField(particles);
		}

		m_NeighborhoodSearch->FindNeighbors();

		TestKernel <<< 1, 3 >>> (particles, d_Info);
		COMPUTE_SAFE(cudaDeviceSynchronize())

		COMPUTE_SAFE(cudaGLUnmapBufferObject(m_VertexBuffer->GetRendererID()))

		// Debug, after the offline solution gets properly implemented this function only needs to be called
		// once after the simulation finishes baking.
		COMPUTE_SAFE(cudaMemcpy(&m_Info, d_Info, sizeof(DFSPHSimulationInfo), cudaMemcpyDeviceToHost))

		m_IterationCount++;
	}

	const Ref<VertexArray>& DFSPHImplementation::GetVertexArray() const
	{
		return m_VertexArray;
	}
}