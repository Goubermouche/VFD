#include "pch.h"
#include "FLIPSimulation.h"

#include "Simulation/FLIP/FLIPSimulation.cuh"
#include "Core/Structures/AxisAlignedBoundingBox.h"
#include "Core/Math/Math.h"
#include "tiny_obj_loader.h"

#include <Glad/glad.h>
#include <cuda_gl_interop.h>

namespace fe {
	void LoadSDFMesh(const std::string& filepath, std::vector<glm::vec3>& vertices, std::vector<glm::ivec3>& triangles) {
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		tinyobj::attrib_t attributes;
		std::vector<float> buffer;

		std::string warning;
		std::string error;

		if (tinyobj::LoadObj(&attributes, &shapes, &materials, &warning, &error, filepath.c_str()) == false) {
			if (error.empty() == false) {
				ERR(error, "triangle mesh");
			}
		}

		for (size_t i = 0; i < attributes.vertices.size(); i += 3)
		{
			vertices.push_back({
				attributes.vertices[i + 0],
				attributes.vertices[i + 1],
				attributes.vertices[i + 2]
			});
		}

		for (size_t i = 0; i < shapes[0].mesh.indices.size(); i += 3)
		{
			triangles.push_back({
				shapes[0].mesh.indices[i + 0].vertex_index,
				shapes[0].mesh.indices[i + 1].vertex_index,
				shapes[0].mesh.indices[i + 2].vertex_index
			});
		}
	}

	FLIPSimulation::FLIPSimulation(const FLIPSimulationDescription& desc)
		: m_Description(desc)
	{
		if (GPUCompute::GetInitState() == false) {
			// The GPU compute context failed to initialize. Return.
			ERR("Simulation stopped (GPU compute context failed to initialize)")
			return;
		}

		float dx = 1.0f / std::max({ desc.Resolution, desc.Resolution, desc.Resolution });

		m_Parameters.TimeStep = desc.TimeStep / desc.SubStepCount;
		m_Parameters.SubStepCount = desc.SubStepCount;
		m_Parameters.Resolution = desc.Resolution;
		m_Parameters.DX = dx;
		m_Parameters.Gravity = { 0.0f, -9.81f, 0.0f };
		m_Parameters.ParticleRadius = (float)(dx * 1.01f * std::sqrt(3.0f) / 2.0f);

		m_MACVelocity.Init(desc.Resolution, desc.Resolution, desc.Resolution, dx);
		m_ValidVelocities.Init(desc.Resolution, desc.Resolution, desc.Resolution);
		m_LiquidSDF.Init(desc.Resolution, desc.Resolution, desc.Resolution, dx);
		m_WeightGrid.Init(desc.Resolution, desc.Resolution, desc.Resolution);
		m_Viscosity.Init(desc.Resolution + 1, desc.Resolution + 1, desc.Resolution + 1, 1.0f);

		// Boundary Mesh
		InitBoundary();
		// AddBoundary();
		AddLiquid("Resources/Models/Polyhedron_1.obj");

		m_Parameters.ParticleCount = m_PositionCache.size();

		m_PositionVAO = Ref<VertexArray>::Create();
		m_PositionVBO = Ref<VertexBuffer>::Create(sizeof(float) * 3 * m_PositionCache.size(), m_PositionCache.data());
		m_PositionVBO->SetLayout({ { ShaderDataType::Float3, "a_Position" } });
		m_PositionVAO->AddVertexBuffer(m_PositionVBO);

		InitMemory();

		LOG("simulation initialized", "FLIP");
	}

	FLIPSimulation::~FLIPSimulation()
	{
	}

	void FLIPSimulation::AddBoundary()
	{
		std::vector<glm::vec3> vertices;
		std::vector<glm::ivec3> triangles;
		LoadSDFMesh("Resources/Models/SphereLarge.obj", vertices, triangles);

		AABB domain({ 0, 0, 0 }, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX);
		AABB bbox(vertices);

		ASSERT(domain.IsPointInside(bbox.GetMinPoint()) && domain.IsPointInside(bbox.GetMaxPoint()), "boundary is not inside the simulation domain! ");

		MeshLevelSet boundarySDF;
		boundarySDF.Init(m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.DX);
		boundarySDF.CalculateSDF(vertices.data(), vertices.size(), triangles.data(), triangles.size(), m_Description.MeshLevelSetExactBand);

		// inverted
		if (true) {
			boundarySDF.Negate();
		}

		// m_SolidSDF.CalculateUnion(boundarySDF);
		LOG("boundary added", "FLIP", ConsoleColor::Cyan);
	}

	// TODO: creates fluid sdfs and unionize them, and then sample them. 
	// TODO: use normalized models 
	void FLIPSimulation::AddLiquid(const std::string& filepath)
	{
		std::vector<glm::vec3> vertices;
		std::vector<glm::ivec3> triangles;
		LoadSDFMesh(filepath, vertices, triangles);

		AABB domain({ 0, 0, 0 }, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX);
		AABB bbox(vertices);

		ASSERT(domain.IsPointInside(bbox.GetMinPoint()) && domain.IsPointInside(bbox.GetMaxPoint()), "fluid is not inside the simulation domain! ");

		MeshLevelSet meshSDF; 

		meshSDF.Init(m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.DX);
		meshSDF.CalculateSDFNew(vertices.data(), vertices.size(), triangles.data(), triangles.size(), m_Description.MeshLevelSetExactBand);

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

					if (meshSDF.TrilinearInterpolate(pos) < 0.0) {
						if (m_SolidSDF.TrilinearInterpolate(pos) >= 0) {
							m_PositionCache.push_back(pos);
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

	TriangleMesh FLIPSimulation::GetBoundaryTriangleMesh()
	{
		float eps = 1e-6;
		AABB domain({ 0, 0, 0 }, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX);
		domain.Expand(-3 * m_Parameters.DX - eps);

		return TriangleMesh(domain);
	}

	void FLIPSimulation::InitBoundary()
	{
		TriangleMesh boundaryMesh = GetBoundaryTriangleMesh();
		m_SolidSDF.Init(m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.DX);
		m_SolidSDF.CalculateSDF(boundaryMesh.GetVertices().data(), boundaryMesh.GetVertexCount(), boundaryMesh.GetTriangles().data(), boundaryMesh.GetTriangleCount(), m_Description.MeshLevelSetExactBand);
		m_SolidSDF.Negate();
		LOG("boundary initialized", "FLIP", ConsoleColor::Cyan);
	}
}