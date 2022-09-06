#include "pch.h"
#include "FLIPSimulation.h"

#include "Simulation/FLIP/FLIPSimulation.cuh"
#include "Core/Structures/AxisAlignedBoundingBox.h"
#include "Core/Math/Math.h"

#include <Glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	FLIPSimulation::FLIPSimulation(const FLIPSimulationDescription& desc)
		: m_Description(desc)
	{
		if (GPUCompute::GetInitState() == false) {
			// The GPU compute context failed to initialize. Return.
			ERR("Simulation stopped (GPU compute context failed to initialize)")
			return;
		}

		float dx = 1.0 / std::max({ desc.Resolution, desc.Resolution, desc.Resolution });

		m_Parameters.SubStepCount = desc.SubStepCount;
		m_Parameters.Resolution = desc.Resolution;
		m_Parameters.DX = dx;
		m_Parameters.Gravity = { 0.0f, -9.81f, 0.0f };
		m_Parameters.ParticleRadius = (float)(dx * 1.01f * std::sqrt(3.0f) / 2.0f);

		m_MACVelocity.Init(desc.Resolution, desc.Resolution, desc.Resolution, dx);
		m_ValidVelocities.Init(desc.Resolution, desc.Resolution, desc.Resolution);
		m_LiquidSDF.Init(desc.Resolution, desc.Resolution, desc.Resolution, dx);
		// m_WeightGrid.Init(desc.Resolution, desc.Resolution, desc.Resolution);
		m_Viscosity.Init(desc.Resolution + 1, desc.Resolution + 1, desc.Resolution + 1, 1.0f);

		// Boundary Mesh
		InitBoundary();
		// AddBoundary("Resources/Models/SDFSafe/Sphere.obj", true);
		AddLiquid("Resources/Models/SDFSafe/Polyhedron_1.obj");
		//AddLiquid("Resources/Models/SDFSafe/Polyhedron_2.obj");
		//AddLiquid("Resources/Models/SDFSafe/Polyhedron_3.obj");
		//AddLiquid("Resources/Models/SDFSafe/Dragon.obj");

		m_Parameters.ParticleCount = m_Particles.size();

		//m_PositionVAO = Ref<VertexArray>::Create();
		//m_PositionVBO = Ref<VertexBuffer>::Create(sizeof(float) * 3 * m_PositionCache.size(), m_PositionCache.data());
		//m_PositionVBO->SetLayout({ { ShaderDataType::Float3, "a_Position" } });
		//m_PositionVAO->AddVertexBuffer(m_PositionVBO);

		InitMemory();

		LOG("simulation initialized", "FLIP");

	}

	FLIPSimulation::~FLIPSimulation()
	{
	}

	void FLIPSimulation::AddBoundary(const std::string& filepath, bool inverted)
	{
		TriangleMesh mesh(filepath);
		MeshLevelSet SDF;
		SDF.Init(mesh, m_Parameters.Resolution, m_Parameters.DX, m_Description.MeshLevelSetExactBand);

		// inverted
		if (inverted) {
			SDF.Negate();
		}

		m_SolidSDF.CalculateUnion(SDF);
		LOG("boundary added", "FLIP", ConsoleColor::Cyan);

		SDF.HostFree();
	}

	// TODO: creates fluid sdfs and unionize them, and then sample them. 
	// TODO: use normalized models 
	// TODO: replace the old particle sampler system with a function utilizing FLIP SDF's 
	void FLIPSimulation::AddLiquid(const std::string& filepath)
	{
		// AABB domain({ 0, 0, 0 }, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX);
		// AABB bbox(vertices);

		// ASSERT(domain.IsPointInside(bbox.GetMinPoint()) && domain.IsPointInside(bbox.GetMaxPoint()), "fluid is not inside the simulation domain! ");

		TriangleMesh mesh(filepath);
		MeshLevelSet SDF;
		SDF.Init(mesh, m_Parameters.Resolution, m_Parameters.DX, m_Description.MeshLevelSetExactBand);

		// Sampling
		uint32_t currentSample = 0;
		uint32_t counterX = 0;
		uint32_t counterY = 0;
		const float diameter = 2.0f * m_Parameters.ParticleRadius;

		float shiftX = std::sqrtf(3.0f) * m_Parameters.ParticleRadius;
		float shiftY = std::sqrtf(6.0f) * diameter / 3.0f;

		// init particles 
		for (int k = 0; k < m_Parameters.Resolution; k++) {
			for (int j = 0; j < m_Parameters.Resolution; j++) {
				for (int i = 0; i < m_Parameters.Resolution; i++) {
					glm::vec3 pos = GridIndexToPosition(i, j, k, m_Parameters.DX);
					glm::vec3 shift = { 0.0f, 0.0f, 0.0f };

					if (counterX % 2)
					{
						shift.z += diameter / (2.0f * (counterY % 2 ? -1 : 1));
					}

					if (counterY % 2)
					{
						shift.x += shiftX / 2.0f;
						shift.z += diameter / 2.0f;
					}

					pos += shift;

					if (SDF.TrilinearInterpolate(pos) < 0.0f) {
						if (m_SolidSDF.TrilinearInterpolate(pos) > 0.0f) {
							m_Particles.push_back({pos});
						}
					}

					currentSample++;
					counterX++;
				}
				counterX = 0;
				counterY++;
			}
			counterY = 0;
		}

		SDF.HostFree();

	 	LOG("liquid added [" + std::to_string(m_Particles.size()) + " particles]", "FLIP", ConsoleColor::Cyan);
	}

	void FLIPSimulation::OnUpdate()
	{
		if (m_Initialized == false || paused) {
			return;
		}
	}

	void FLIPSimulation::OnRenderTemp()
	{
		for (size_t i = 0; i < m_Particles.size(); i++)
		{
			Renderer::DrawPoint(m_Particles[i].Position, { 0.7f, 0.7f, 0.7f,1 }, 0.3f);
		}
	}

	void FLIPSimulation::InitMemory()
	{
		//m_MACVelocityDevice = m_MACVelocity.UploadToDevice();

		//FLIPUploadMACVelocitiesToSymbol(m_MACVelocityDevice);
		//FLIPUploadSimulationParametersToSymbol(m_Parameters);

		m_Initialized = true;
	}

	void FLIPSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

		m_MACVelocity.HostFree();
		m_WeightGrid.HostFree();
		m_SolidSDF.HostFree();
		m_LiquidSDF.HostFree();
		m_SavedVelocityField.HostFree();
	}

	TriangleMesh FLIPSimulation::GetBoundaryTriangleMesh()
	{
		AABB domain({ 0, 0, 0 }, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX);
		domain.Expand(-3 * m_Parameters.DX);

		return TriangleMesh(domain );
	}

	void FLIPSimulation::InitBoundary()
	{ 
		TriangleMesh boundaryMesh = GetBoundaryTriangleMesh();
		m_SolidSDF.Init(boundaryMesh, m_Parameters.Resolution, m_Parameters.DX, m_Description.MeshLevelSetExactBand);
		m_SolidSDF.Negate();
		LOG("boundary initialized", "FLIP", ConsoleColor::Cyan);
	}
}