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

		float dx = 1.0f / std::max({ desc.Size.x, desc.Size.y, desc.Size.z });

		m_Parameters.TimeStep = desc.TimeStep / desc.SubStepCount;
		m_Parameters.SubStepCount = desc.SubStepCount;
		m_Parameters.Size = desc.Size;
		m_Parameters.DX = dx;
		m_Parameters.Gravity = { 0.0f, -9.81f, 0.0f };
		m_Parameters.ParticleRadius = (float)(dx * 1.01f * std::sqrt(3.0f) / 2.0f);

		m_MACVelocity.Init(desc.Size.x, desc.Size.y, desc.Size.z, dx);
		m_ValidVelocities.Init(desc.Size.x, desc.Size.y, desc.Size.z);
		m_LiquidSDF.Init(desc.Size.x, desc.Size.y, desc.Size.z, dx);
		m_WeightGrid.Init(desc.Size.x, desc.Size.y, desc.Size.z);
		m_Viscosity.Init(desc.Size.x + 1, desc.Size.y + 1, desc.Size.z + 1, 1.0f);

		// Boundary Mesh
		InitBoundary();
		AddBoundary();
		AddLiquid();

		m_Parameters.ParticleCount = m_PositionCache.size();

		m_PositionVAO = Ref<VertexArray>::Create();
		m_PositionVBO = Ref<VertexBuffer>::Create(sizeof(float) * 3 * m_PositionCache.size(), m_PositionCache.data());
		m_PositionVBO->SetLayout({ { ShaderDataType::Float4, "a_Position" } });
		m_PositionVAO->AddVertexBuffer(m_PositionVBO);

		InitMemory();

		LOG("simulation initialized", "FLIP");
	}

	FLIPSimulation::~FLIPSimulation()
	{
	}

	void FLIPSimulation::AddBoundary()
	{
		TriangleMesh mesh("Resources/Models/SphereLarge.obj");

		AABB domain({ 0, 0, 0 }, m_Parameters.Size.x * m_Parameters.DX, m_Parameters.Size.y * m_Parameters.DX, m_Parameters.Size.z * m_Parameters.DX);
		AABB bbox(mesh.GetVertices());

		ASSERT(domain.IsPointInside(bbox.GetMinPoint()) && domain.IsPointInside(bbox.GetMaxPoint()), "boundary is not inside the simulation domain! ");

		MeshLevelSet boundarySDF;
		boundarySDF.Init(m_Parameters.Size.x, m_Parameters.Size.y, m_Parameters.Size.z, m_Parameters.DX);
		boundarySDF.CalculateSDF(mesh.GetVertices().data(), mesh.GetVertexCount(), mesh.GetTriangles().data(), mesh.GetTriangleCount(), m_Description.MeshLevelSetExactBand);

		// inverted
		if (true) {
			boundarySDF.Negate();
		}

		m_SolidSDF.CalculateUnion(boundarySDF);
		LOG("boundary added", "FLIP", ConsoleColor::Cyan);
	}

	void FLIPSimulation::AddLiquid()
	{
		TriangleMesh mesh("Resources/Models/Bunny_2.obj");

		AABB domain({ 0, 0, 0 }, m_Parameters.Size.x * m_Parameters.DX, m_Parameters.Size.y * m_Parameters.DX, m_Parameters.Size.z * m_Parameters.DX);
		AABB bbox(mesh.GetVertices());

		ASSERT(domain.IsPointInside(bbox.GetMinPoint()) && domain.IsPointInside(bbox.GetMaxPoint()), "fluid is not inside the simulation domain! ");

		MeshLevelSet meshSDF; 
		meshSDF.Init(m_Parameters.Size.x, m_Parameters.Size.y, m_Parameters.Size.z, m_Parameters.DX);
		meshSDF.CalculateSDF(mesh.GetVertices().data(), mesh.GetVertexCount(), mesh.GetTriangles().data(), mesh.GetTriangleCount(), m_Description.MeshLevelSetExactBand);

		// init particles 
		for (int k = 0; k < m_Parameters.Size.z; k++) {
			for (int j = 0; j < m_Parameters.Size.y; j++) {
				for (int i = 0; i < m_Parameters.Size.x; i++) {
					glm::vec3 gpos = GridIndexToPosition(i, j, k, m_Parameters.DX);

					for (int i_dx = 0; i_dx < 8; i_dx++) {
						float a = Random::RandomFloat(0.0f, m_Parameters.DX);
						float b = Random::RandomFloat(0.0f, m_Parameters.DX);
						float c = Random::RandomFloat(0.0f, m_Parameters.DX);
						glm::vec3 jitter = { a, b, c };
						glm::vec3 pos = gpos + jitter;

						if (meshSDF.TrilinearInterpolate(pos) < 0.0) {
							float solid_phi = m_SolidSDF.TrilinearInterpolate(pos);
							if (solid_phi >= 0) {
								m_PositionCache.push_back(pos);
							}
						}
					}
				}
			}
		}

		LOG("liquid added [" + std::to_string(m_PositionCache.size()) + " particles]", "FLIP", ConsoleColor::Cyan);
	}

	void FLIPSimulation::OnUpdate()
	{
		if (m_Initialized == false || paused) {
			return;
		}
	}

	void FLIPSimulation::OnRenderTemp()
	{
		for (size_t i = 0; i < m_PositionCache.size(); i++)
		{
			Renderer::DrawPoint(m_PositionCache[i], { 1, 1, 1,1 }, 0.1f);
		}
	}

	void FLIPSimulation::InitMemory()
	{
		m_MACVelocityDevice = m_MACVelocity.UploadToDevice();

		FLIPUploadMACVelocitiesToSymbol(m_MACVelocityDevice);
		FLIPUploadSimulationParametersToSymbol(m_Parameters);

		m_Initialized = true;

		// TEMP test kernel
		FLIPUpdateFluidSDF();
	}

	void FLIPSimulation::FreeMemory()
	{
		if (m_Initialized == false) {
			return;
		}

		m_MACVelocityDevice.Free();
	}

	//TriangleMesh FLIPSimulation::GetBoundaryTriangleMesh()
	//{
	//	float eps = 1e-6;
	//	AABB domain({ 0, 0, 0 }, m_Parameters.Size.x * m_Parameters.DX, m_Parameters.Size.y * m_Parameters.DX, m_Parameters.Size.z * m_Parameters.DX);
	//	domain.Expand(-3 * m_Parameters.DX - eps);

	//}

	TriangleMesh FLIPSimulation::GetBoundaryTriangleMesh()
	{
		float eps = 1e-6;
		AABB domain({ 0, 0, 0 }, m_Parameters.Size.x * m_Parameters.DX, m_Parameters.Size.y * m_Parameters.DX, m_Parameters.Size.z * m_Parameters.DX);
		domain.Expand(-3 * m_Parameters.DX - eps);

		return TriangleMesh(domain);
	}

	void FLIPSimulation::InitBoundary()
	{
		TriangleMesh boundaryMesh = GetBoundaryTriangleMesh();
		m_SolidSDF.Init(m_Parameters.Size.x, m_Parameters.Size.y, m_Parameters.Size.z, m_Parameters.DX);
		m_SolidSDF.CalculateSDF(boundaryMesh.GetVertices().data(), boundaryMesh.GetVertexCount(), boundaryMesh.GetTriangles().data(), boundaryMesh.GetTriangleCount(), m_Description.MeshLevelSetExactBand);
		m_SolidSDF.Negate();
		LOG("boundary initialized", "FLIP", ConsoleColor::Cyan);
	}
}