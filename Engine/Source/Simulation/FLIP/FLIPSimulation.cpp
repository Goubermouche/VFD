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

		float dx = 1.0f / std::max({ desc.Resolution, desc.Resolution, desc.Resolution });

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
		// AddBoundary("Resources/Models/SphereLarge.obj", true);
		AddLiquid("Resources/Models/SDFSafe/Polyhedron_1.obj");
		//AddLiquid("Resources/Models/SDFSafe/Polyhedron_2.obj");
		//AddLiquid("Resources/Models/SDFSafe/Polyhedron_3.obj");
		//AddLiquid("Resources/Models/SDFSafe/Dragon.obj");

		m_Parameters.ParticleCount = m_Particles.size();

		//m_PositionVAO = Ref<VertexArray>::Create();
		//m_PositionVBO = Ref<VertexBuffer>::Create(sizeof(float) * 3 * m_PositionCache.size(), m_PositionCache.data());
		//m_PositionVBO->SetLayout({ { ShaderDataType::Float3, "a_Position" } });
		//m_PositionVAO->AddVertexBuffer(m_PositionVBO);

		SetViscosity(desc.Viscosity);


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

	void FLIPSimulation::SetViscosity(float value)
	{
		ASSERT(value >= 0.0f, "viscosity cannot be negative!");
		m_Parameters.Viscosity = value;
		for (int k = 0; k < m_Viscosity.Size.z; k++) {
			for (int j = 0; j < m_Viscosity.Size.y; j++) {
				for (int i = 0; i < m_Viscosity.Size.x; i++) {
					m_Viscosity.Set(i, j, k, value);
				}
			}
		}
	}

	void FLIPSimulation::OnUpdate()
	{
		if (m_Initialized == false || paused) {
			return;
		}

		float t = 0.0f;
		while (t < m_Description.TimeStep) {
			float subStep = CFL();
			if (t + subStep > m_Description.TimeStep) {
				subStep = m_Description.TimeStep - t;
			}

			UpdateFluidSDF();
			AdvectVelocityField();
			AddBodyForce(subStep);

			ERR("update");
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

		m_MACVelocityDevice.HostFree();
		m_MACVelocity.HostFree();
		m_WeightGrid.HostFree();
		m_SolidSDF.HostFree();
		m_LiquidSDF.HostFree();
	}

	TriangleMesh FLIPSimulation::GetBoundaryTriangleMesh()
	{
		AABB domain({ 0, 0, 0 }, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX, m_Parameters.Resolution * m_Parameters.DX);
		domain.Expand(-3 * m_Parameters.DX);

		return TriangleMesh(domain);
	}

	void FLIPSimulation::InitBoundary()
	{ 
		TriangleMesh boundaryMesh = GetBoundaryTriangleMesh();
		m_SolidSDF.Init(boundaryMesh, m_Parameters.Resolution, m_Parameters.DX, m_Description.MeshLevelSetExactBand);
		m_SolidSDF.Negate();
		LOG("boundary initialized", "FLIP", ConsoleColor::Cyan);
	}

	float FLIPSimulation::CFL()
	{
		float maxVel = 0.0f;
		for (int k = 0; k < m_Parameters.Resolution; k++) {
			for (int j = 0; j < m_Parameters.Resolution; j++) {
				for (int i = 0; i < m_Parameters.Resolution + 1; i++) {
					maxVel = fmax(maxVel, fabs(m_MACVelocity.U(i, j, k)));
				}
			}
		}

		for (int k = 0; k < m_Parameters.Resolution; k++) {
			for (int j = 0; j < m_Parameters.Resolution + 1; j++) {
				for (int i = 0; i < m_Parameters.Resolution; i++) {
					maxVel = fmax(maxVel, fabs(m_MACVelocity.V(i, j, k)));
				}
			}
		}

		for (int k = 0; k < m_Parameters.Resolution + 1; k++) {
			for (int j = 0; j < m_Parameters.Resolution; j++) {
				for (int i = 0; i < m_Parameters.Resolution; i++) {
					maxVel = fmax(maxVel, fabs(m_MACVelocity.W(i, j, k)));
				}
			}
		}

		return (float)((m_Description.CFLConditionNumber * m_Parameters.DX) / maxVel);
	}

	void FLIPSimulation::UpdateFluidSDF()
	{
		std::vector<glm::vec3> points;
		points.reserve(m_Particles.size());
		for (size_t i = 0; i < m_Particles.size(); i++) {
			points.push_back(m_Particles[i].Position);
		}

		m_LiquidSDF.CalculateSDF(points, m_Parameters.ParticleRadius, m_SolidSDF);
	}

	void FLIPSimulation::ComputeVelocityScalarField(Array3D<float>& field, Array3D<bool>& isValueSet, int dir)
	{
		int U = 0; int V = 1; int W = 2;
		glm::vec3 offset;
		float hdx = (float)(0.5f * m_Parameters.DX);

		if (dir == U) {
			offset = glm::vec3(0.0f, hdx, hdx);
		}
		else if (dir == V) {
			offset = glm::vec3(hdx, 0.0f, hdx);
		}
		else if (dir == W) {
			offset = glm::vec3(hdx, hdx, 0.0f);
		}
		else {
			return;
		}

		Array3D<float> weights;
		weights.Init(field.Size.x, field.Size.y, field.Size.z, 0.0f);

		// coefficients for Wyvill kernel
		float r = m_Parameters.DX;
		float rsq = r * r;
		float coef1 = (4.0f / 9.0f) * (1.0f / (r * r * r * r * r * r));
		float coef2 = (17.0f / 9.0f) * (1.0f / (r * r * r * r));
		float coef3 = (22.0f / 9.0f) * (1.0f / (r * r));

		// transfer particle velocity component to grid
		for (size_t pidx = 0; pidx < m_Particles.size(); pidx++) {
			glm::vec3 p = m_Particles[pidx].Position - offset;
			float velocityComponent = m_Particles[pidx].Velocity[dir];

			glm::ivec3 g = PositionToGridIndex(p, r);
			glm::ivec3 gmin((int)fmax(g.x - 1, 0),
				(int)fmax(g.y - 1, 0),
				(int)fmax(g.z - 1, 0));
			glm::ivec3 gmax((int)fmin(g.x + 1, field.Size.x - 1),
				(int)fmin(g.y + 1, field.Size.y - 1),
				(int)fmin(g.z + 1, field.Size.z - 1));

			for (int k = gmin.z; k <= gmax.z; k++) {
				for (int j = gmin.y; j <= gmax.y; j++) {
					for (int i = gmin.x; i <= gmax.x; i++) {

						glm::vec3 gpos = GridIndexToPosition(i, j, k, r);
						glm::vec3 v = gpos - p;
						float distsq = glm::dot(v, v);
						if (distsq < rsq) {
							float weight = 1.0f - coef1 * distsq * distsq * distsq +
								coef2 * distsq * distsq -
								coef3 * distsq;
							field.Add(i, j, k, weight * velocityComponent);
							weights.Add(i, j, k, weight);
						}
					}
				}
			}
		}

		// Divide field values by weights
		double eps = 1e-9;
		for (int k = 0; k < field.Size.z; k++) {
			for (int j = 0; j < field.Size.y; j++) {
				for (int i = 0; i < field.Size.x; i++) {
					float value = field(i, j, k);
					float weight = weights(i, j, k);

					if (weight < eps) {
						continue;
					}
					field.Set(i, j, k, value / weight);
					isValueSet.Set(i, j, k, true);
				}
			}
		}

		weights.HostFree();
	}

	void FLIPSimulation::AdvectVelocityField()
	{
		Array3D<bool> fluidCellGrid;
		fluidCellGrid.Init(m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.Resolution, false);
		for (int k = 0; k < m_Parameters.Resolution; k++) {
			for (int j = 0; j < m_Parameters.Resolution; j++) {
				for (int i = 0; i < m_Parameters.Resolution; i++) {
					if (m_LiquidSDF(i, j, k) < 0.0) {
						fluidCellGrid.Set(i, j, k, true);
					}
				}
			}
		}

		m_ValidVelocities.Reset();
		AdvectVelocityFieldU(fluidCellGrid);
		AdvectVelocityFieldV(fluidCellGrid);
		AdvectVelocityFieldW(fluidCellGrid);

		fluidCellGrid.HostFree();
	}

	void FLIPSimulation::AddBodyForce(float dt)
	{
		Array3D<bool> fgrid;
		fgrid.Init(m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.Resolution, false);

		for (int k = 0; k < m_Parameters.Resolution; k++) {
			for (int j = 0; j < m_Parameters.Resolution; j++) {
				for (int i = 0; i < m_Parameters.Resolution; i++) {
					if (m_LiquidSDF(i, j, k) < 0.0) {
						fgrid.Set(i, j, k, true);
					}
				}
			}
		}

		for (int k = 0; k < m_Parameters.Resolution; k++) {
			for (int j = 0; j < m_Parameters.Resolution; j++) {
				for (int i = 0; i < m_Parameters.Resolution + 1; i++) {
					if (IsFaceBorderingValueU(i, j, k, true, fgrid)) {
						m_MACVelocity.AddU(i, j, k, m_Parameters.Gravity.x * dt);
					}
				}
			}
		}

		for (int k = 0; k < m_Parameters.Resolution; k++) {
			for (int j = 0; j < m_Parameters.Resolution + 1; j++) {
				for (int i = 0; i < m_Parameters.Resolution; i++) {
					if (IsFaceBorderingValueV(i, j, k, true, fgrid)) {
						m_MACVelocity.AddV(i, j, k, m_Parameters.Gravity.y * dt);
					}
				}
			}
		}

		for (int k = 0; k < m_Parameters.Resolution + 1; k++) {
			for (int j = 0; j < m_Parameters.Resolution; j++) {
				for (int i = 0; i < m_Parameters.Resolution; i++) {
					if (IsFaceBorderingValueW(i, j, k, true, fgrid)) {
						m_MACVelocity.AddW(i, j, k, m_Parameters.Gravity.z * dt);
					}
				}
			}
		}

		fgrid.HostFree();
	}

	void FLIPSimulation::AdvectVelocityFieldU(Array3D<bool>& fluidCellGrid)
	{
		Array3D<float> ugrid;
		Array3D<bool> isValueSet;
		ugrid.Init(m_Parameters.Resolution + 1, m_Parameters.Resolution, m_Parameters.Resolution, 0.0f);
		isValueSet.Init(m_Parameters.Resolution + 1, m_Parameters.Resolution, m_Parameters.Resolution, false);
		ComputeVelocityScalarField(ugrid, isValueSet, 0);

		m_MACVelocity.ClearU();
		for (int k = 0; k < ugrid.Size.z; k++) {
			for (int j = 0; j < ugrid.Size.y; j++) {
				for (int i = 0; i < ugrid.Size.x; i++) {
					if (IsFaceBorderingValueU(i, j, k, true, fluidCellGrid)) {
						if (isValueSet(i, j, k)) {
							m_MACVelocity.SetU(i, j, k, ugrid(i, j, k));
							m_ValidVelocities.ValidU.Set(i, j, k, true);
						}
					}
				}
			}
		}

		ugrid.HostFree();
		isValueSet.HostFree();
	}

	void FLIPSimulation::AdvectVelocityFieldV(Array3D<bool>& fluidCellGrid)
	{
		Array3D<float> vgrid;
		Array3D<bool> isValueSet;
		vgrid.Init(m_Parameters.Resolution, m_Parameters.Resolution + 1, m_Parameters.Resolution, 0.0f);
		isValueSet.Init(m_Parameters.Resolution, m_Parameters.Resolution + 1, m_Parameters.Resolution, false);
		ComputeVelocityScalarField(vgrid, isValueSet, 1);

		m_MACVelocity.ClearV();
		for (int k = 0; k < vgrid.Size.z; k++) {
			for (int j = 0; j < vgrid.Size.y; j++) {
				for (int i = 0; i < vgrid.Size.x; i++) {
					if (IsFaceBorderingValueV(i, j, k, true, fluidCellGrid)) {
						if (isValueSet(i, j, k)) {
							 m_MACVelocity.SetV(i, j, k, vgrid(i, j, k));
							 m_ValidVelocities.ValidV.Set(i, j, k, true);
						}
					}
				}
			}
		}

		vgrid.HostFree();
		isValueSet.HostFree();
	}

	void FLIPSimulation::AdvectVelocityFieldW(Array3D<bool>& fluidCellGrid)
	{
		Array3D<float> wgrid;
		Array3D<bool> isValueSet;
		wgrid.Init(m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.Resolution + 1, 0.0f);
		isValueSet.Init(m_Parameters.Resolution, m_Parameters.Resolution, m_Parameters.Resolution + 1, false);
		ComputeVelocityScalarField(wgrid, isValueSet, 2);

		m_MACVelocity.ClearW();
		for (int k = 0; k < wgrid.Size.z; k++) {
			for (int j = 0; j < wgrid.Size.y; j++) {
				for (int i = 0; i < wgrid.Size.x; i++) {
					if (IsFaceBorderingValueW(i, j, k, true, fluidCellGrid)) {
						if (isValueSet(i, j, k)) {
							m_MACVelocity.SetW(i, j, k, wgrid(i, j, k));
							m_ValidVelocities.ValidW.Set(i, j, k, true);
						}
					}
				}
			}
		}

		wgrid.HostFree();
		isValueSet.HostFree();
	}
}