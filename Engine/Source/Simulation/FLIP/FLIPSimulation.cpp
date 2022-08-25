#include "pch.h"
#include "FLIPSimulation.h"

#include "Simulation/FLIP/FLIPSimulation.cuh"
#include "Core/Structures/AxisAlignedBoundingBox.h"

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

	void FLIPSimulation::OnUpdate()
	{
		if (m_Initialized == false || paused) {
			return;
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