#include "pch.h"
#include "DFSPHSimulation.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Utility/SDF/MeshDistance.h"
#include "Core/Math/GaussQuadrature.h"
#include "Core/Math/HaltonVec323.h"

#define forall_fluid_neighbors(code) \
	for (unsigned int j = 0; j < NumberOfNeighbors(0, 0, i); j++) \
	{ \
		const unsigned int neighborIndex = GetNeighbor(0, 0, i, j); \
		const glm::vec3 &xj = m_ParticlePositions[neighborIndex]; \
		code \
	} \

#define forall_volume_maps(code) \
	for(unsigned int pid = 0; pid < m_RigidBodies.size(); pid++) { \
		StaticRigidBody* bm_neighbor = m_RigidBodies[pid]; \
		const float Vj = bm_neighbor->GetBoundaryVolume(i);  \
		if (Vj > 0.0) \
		{ \
			const glm::vec3 &xj = bm_neighbor->GetBoundaryXJ(i); \
			code \
		} \
	} \
	

#define forall_fluid_neighbors_avx(code)\
	const unsigned int maxN = m_Base->NumberOfNeighbors(0, 0, i); \
	for (unsigned int j = 0; j < maxN; j += 8) \
	{ \
		const unsigned int count = std::min(maxN - j, 8u); \
		const Scalar3f8 xj_avx = ConvertScalarZero(&m_Base->GetNeighborList(0, 0, i)[j], &m_ParticlePositions[0], count); \
		code \
	} \

#define forall_fluid_neighbors_in_same_phase(code) \
	for (unsigned int j = 0; j < m_Base->NumberOfNeighbors(0, 0, i); j++) \
	{ \
		const unsigned int neighborIndex = m_Base->GetNeighbor(0, 0, i, j); \
		const glm::vec3 &xj = m_Base->GetParticlePosition(neighborIndex); \
		code \
	} 

#define forall_fluid_neighbors_avx_nox(code) \
	unsigned int idx = 0; \
	const unsigned int maxN = m_Base->NumberOfNeighbors(0, 0, i); \
	for (unsigned int j = 0; j < maxN; j += 8) \
	{ \
		const unsigned int count = std::min(maxN - j, 8u); \
		code \
		idx++; \
	} \

#define forall_fluid_neighbors_in_same_phase_avx(code) \
    const unsigned int maxN = sim->NumberOfNeighbors(0, 0, i);  \
    for (unsigned int j = 0; j < maxN; j += 8) \
    { \
		const unsigned int count = std::min(maxN - j, 8u); \
		const Scalar3f8 xj_avx = ConvertScalarZero(&sim->GetNeighborList(0, 0, i)[j], &sim->GetParticlePosition(0), count); \
		code \
	} \

#define compute_Vj_gradW() const Scalar3f8& V_gradW = m_PrecalculatedVolumeGradientW[m_PrecalculatedIndices[i] + idx];
#define compute_Vj_gradW_samephase() const Scalar3f8& V_gradW = sim->GetPrecalculatedVolumeGradientW(sim->GetPrecalculatedIndicesSamePhase(i) + j / 8);

namespace fe {
	DFSPHSimulation::DFSPHSimulation(const DFSPHSimulationDescription& desc)
	{
		m_Material = Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl"));
		m_Material->Set("color", { 0.4f, 0.4f, 0.4f });

		// Init the scene 
		m_TimeStepSize = 0.001f;
		m_ParticleRadius = 0.025f;

		// Init sim 
		{
			m_Gravity = { 0.0f, -9.81f, 0.0f };
			// m_gravitation = { 0.0, 0.0, 0.0 };
			m_CFLFactor = 1.0f;
			m_CFLMinTimeStepSize = 0.0001f;
			m_CFLMaxTimeStepSize = 0.005f;

			SetParticleRadius(m_ParticleRadius);

			m_NeighborhoodSearch = new NeighborhoodSearch(m_SupportRadius, false);
			m_NeighborhoodSearch->SetRadius(m_SupportRadius);
		}

		m_WZero = PrecomputedCubicKernel::WZero();
		InitFluidData();

		{
			m_Factor.resize(m_ParticleCount, 0.0);
			m_Kappa.resize(m_ParticleCount, 0.0);
			m_KappaVolume.resize(m_ParticleCount, 0.0);
			m_DensityAdvection.resize(m_ParticleCount, 0.0);
		}

		{
			StaticRigidBodyDescription data;
			data.SourceMesh = "Resources/Models/Cube.obj";
			data.Position = { 0, 3, 0 };
			data.Rotation = glm::angleAxis(glm::radians(0.0f), glm::vec3(0.f, 0.f, 0.f));
			data.Scale = { 2, 0.25, 2 };
			data.Inverted = false;
			data.Padding = 0.0;
			data.CollisionMapResolution = { 20, 20, 20 };

			StaticRigidBody* rb = new StaticRigidBody(data, this);
			m_RigidBodies.push_back(rb);
		}

		{
			StaticRigidBodyDescription data;
			data.SourceMesh = "Resources/Models/Torus.obj";
			data.Position = { 0, 4, 0 };
			data.Rotation = glm::angleAxis(glm::radians(0.0f), glm::vec3(0.f, 0.f, 0.f));
			data.Scale = { .5, .5, .5 };
			data.Inverted = false;
			data.Padding = 0.0;
			data.CollisionMapResolution = { 20, 20, 20 };

			StaticRigidBody* rb = new StaticRigidBody(data, this);
			m_RigidBodies.push_back(rb);
		}

		m_SurfaceTensionSolver = new SurfaceTensionZorillaRitter2020(this);
		m_ViscositySolver = new ViscosityWeiler2018(this);
	}

	DFSPHSimulation::~DFSPHSimulation()
	{
	}

	void DFSPHSimulation::OnUpdate()
	{
		if (paused) {
			return;
		}

		const float h = m_TimeStepSize;
		{
			if (m_FrameCounter % 500 == 0) {
				const unsigned int numPart = m_ParticleCount;

				{
					m_NeighborhoodSearch->ZSort();

					if (numPart > 0) {
						auto const& d = m_NeighborhoodSearch->GetPointSet(0);

						d.SortField(&m_ParticlePositions[0]);
						d.SortField(&m_ParticleVelocities[0]);
						d.SortField(&m_ParticleAccelerations[0]);
						d.SortField(&m_ParticleMasses[0]);
						d.SortField(&m_ParticleDensities[0]);

						// Viscosity
						d.SortField(&m_ViscositySolver->m_vDiff[0]);

						// Surface tension
						d.SortField(&m_SurfaceTensionSolver->m_mc_normals[0]);
						d.SortField(&m_SurfaceTensionSolver->m_final_curvatures[0]);
						d.SortField(&m_SurfaceTensionSolver->m_pca_curv[0]);
						d.SortField(&m_SurfaceTensionSolver->m_pca_curv_smooth[0]);
						d.SortField(&m_SurfaceTensionSolver->m_mc_curv[0]);
						d.SortField(&m_SurfaceTensionSolver->m_mc_curv_smooth[0]);
						d.SortField(&m_SurfaceTensionSolver->m_mc_normals_smooth[0]);
						d.SortField(&m_SurfaceTensionSolver->m_pca_normals[0]);
						d.SortField(&m_SurfaceTensionSolver->m_final_curvatures_old[0]);
						d.SortField(&m_SurfaceTensionSolver->m_classifier_input[0]);
						d.SortField(&m_SurfaceTensionSolver->m_classifier_output[0]);
					}
				}
			
				// m_simulationData.performNeighborhoodSearchSort();
				if (numPart > 0)
				{
					auto const& d = m_NeighborhoodSearch->GetPointSet(0);
					d.SortField(&m_Kappa[0]);
					d.SortField(&m_KappaVolume[0]);
				}
			}

			m_FrameCounter++;
			m_NeighborhoodSearch->FindNeighbors();
		}

		PrecomputeValues();

		ComputeVolumeAndBoundaryX();

		ComputeDensities();

		ComputeDFSPHFactor();

		DivergenceSolve();

		ClearAccelerations();

		// Non-Pressure forces
		// m_surfaceTension->OnUpdate();
		m_ViscositySolver->OnUpdate();

		UpdateTimeStepSize();

		{
			const unsigned int numParticles = m_ParticleCount;
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++) {
					glm::vec3& vel = m_ParticleVelocities[i];
					vel += h * m_ParticleAccelerations[i];
				}
			}
		}

		PressureSolve();

		{
			const unsigned int numParticles = m_ParticleCount;
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					glm::vec3& xi = m_ParticlePositions[i];
					const glm::vec3& vi = m_ParticleVelocities[i];
					xi += h * vi;
				}
			}
		}
	}

	void DFSPHSimulation::OnRenderTemp()
	{
		for (size_t pid = 0; pid < m_RigidBodies.size(); pid++)
		{
			const StaticRigidBodyDescription& desc = m_RigidBodies[pid]->GetDescription();

			glm::mat4 t = glm::translate(glm::mat4(1.0f), { desc.Position.x, desc.Position.y - 0.25f, desc.Position.z });
			// t = glm::scale(t, { 1.0f, 1.0f, 1.0f });
			t = t * glm::toMat4(desc.Rotation);

			m_Material->Set("model", t);

			Renderer::DrawTriangles(m_RigidBodies[pid]->GetGeometry().GetVAO(), m_RigidBodies[pid]->GetGeometry().GetVertexCount(), m_Material);
		}
	
		for (size_t i = 0; i < m_ParticleCount; i++)
		{
			// default
			Renderer::DrawPoint(m_ParticlePositions[i], { 0.65f, 0.65f, 0.65f, 1 }, m_ParticleRadius * 35);

			// density
			// float v = m_density[i] / 1000.0f;
			// Renderer::DrawPoint(m_x[i] + offset, { v, 0, v, 1}, particleRadius * 35);

			// factor
			// float v = 1.0f + m_simulationData.GetFactor(0, i) * 500;
			// Renderer::DrawPoint(m_x[i] + offset, { v, 0, v, 1 }, particleRadius * 35);

			// Kappa
			// float v = -m_simulationData.GetKappa(0, i) * 10000;
			// Renderer::DrawPoint(m_x[i] + offset, { v, 0, v, 1 }, particleRadius * 35);

			// Kappa V
			// float v = -m_simulationData.GetKappaV(0, i) * 10;
			// Renderer::DrawPoint(m_x[i] + offset, { v, 0, v, 1 }, particleRadius * 35);

			//float v = m_surfaceTension->m_classifier_output[i];
			//Renderer::DrawPoint(m_x[i] + offset, { v, 0, v, 1 }, particleRadius * 35);
		}

		//{

		//	glm::vec3 size = { 5, 10, 5 };
		//	float step = 0.5f;
		//	for (float x =-size.x; x < size.x; x+= step)
		//	{
		//		for (float y = -size.y; y < size.y; y+= step)
		//		{
		//			for (float z = -size.z; z < size.z; z+= step)
		//			{
		//				glm::vec3 point = { x, y, z };

		//				double dist = m_boundaryModels->m_map->Interpolate(1, point);

		//				// inside
		//				if (dist > 0.0) {
		//					//Renderer::DrawPoint(point, { 1.0f, 0.0f, 0.0f, 1.0f }, 1);
		//				}
		//				else {
		//					Renderer::DrawPoint(point, { 0.0f, 1.0f, 0.0f, 1.0f }, 1);
		//				}
		//			}
		//		}
		//	}
		//}
	}

	void DFSPHSimulation::UpdateVMVelocity()
	{

	}

	void DFSPHSimulation::SetParticleRadius(float val)
	{
		m_SupportRadius = static_cast<float>(4.0) * val;
		PrecomputedCubicKernel::SetRadius(m_SupportRadius);
		CubicKernelAVX::SetRadius(m_SupportRadius);
	}

	void DFSPHSimulation::ComputeVolumeAndBoundaryX()
	{
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < m_ParticleCount; i++)
			{
				const glm::vec3& xi = m_ParticlePositions[i];
				ComputeVolumeAndBoundaryX(i, xi);
			}
		}
	}

	void DFSPHSimulation::ComputeVolumeAndBoundaryX(const unsigned int i, const glm::vec3& xi)
	{

		for (unsigned int pid = 0; pid < m_RigidBodies.size(); pid++)
		{
			StaticRigidBody* bm = m_RigidBodies[pid];
			glm::vec3& boundaryXj = bm->GetBoundaryXJ(i);
			boundaryXj = { 0.0, 0.0, 0.0 };
			float& boundaryVolume = bm->GetBoundaryVolume(i);
			boundaryVolume = 0.0;

			const glm::vec3& t = { 0, -0.25, 0 };
			glm::mat3 R = glm::toMat3(bm->GetRotation());

			glm::dvec3 normal;
			const glm::dvec3 localXi = (glm::transpose(R) * ((glm::dvec3)xi - (glm::dvec3)t));

			std::array<unsigned int, 32> cell;
			glm::dvec3 c0;
			std::array<double, 32> N;
			std::array<std::array<double, 3>, 32> dN;
			bool chk = bm->GetCollisionMap()->DetermineShapeFunctions(0, localXi, cell, c0, N, &dN);

			double dist = std::numeric_limits<double>::max();
			if (chk) {
				dist = bm->GetCollisionMap()->Interpolate(0, localXi, cell, c0, N, &normal, &dN);
			}

			bool animateParticle = false;

			if ((dist > 0.0) && (static_cast<float>(dist) < m_SupportRadius))
			{
				const double volume = bm->GetCollisionMap()->Interpolate(1, localXi, cell, c0, N);
				if ((volume > 0.0) && (volume != std::numeric_limits<double>::max()))
				{
					boundaryVolume = static_cast<float>(volume);

					normal = R * normal;
					const double nl = std::sqrt(glm::dot(normal, normal));

					if (nl > 1.0e-9)
					{
						normal /= nl;
						const float d = std::max((static_cast<float>(dist) + static_cast<float>(0.5) * m_ParticleRadius), static_cast<float>(2.0) * m_ParticleRadius);
						boundaryXj = (xi - d * (glm::vec3)normal);
					}
					else
					{
						boundaryVolume = 0.0;
					}
				}
				else
				{
					boundaryVolume = 0.0;
				}
			}
			else if (dist <= 0.0)
			{
				animateParticle = true;
				boundaryVolume = 0.0;
			}
			else
			{
				boundaryVolume = 0.0;
			}

			if (animateParticle)
			{
				if (dist != std::numeric_limits<double>::max())				// if dist is numeric_limits<double>::max(), then the particle is not close to the current boundary
				{
					normal = R * normal;
					const double nl = std::sqrt(glm::dot(normal, normal));

					if (nl > 1.0e-5)
					{
						normal /= nl;
						// project to surface
						float delta = static_cast<float>(2.0) * m_ParticleRadius - static_cast<float>(dist);
						delta = std::min(delta, static_cast<float>(0.1) * m_ParticleRadius);		// get up in small steps
						m_ParticlePositions[i] = (xi + delta * (glm::vec3)normal);
						// adapt velocity in normal direction
						// m_v[i] = 1.0 / timeStepSize * delta * normal;
						m_ParticleVelocities[i] = { 0.0, 0.0, 0.0 };
					}
				}
				boundaryVolume = 0.0;
			}
		}
	}

	void DFSPHSimulation::ComputeDensities()
	{
		const float density0 = m_Density0;
		const unsigned int numParticles = m_ParticleCount;
		auto* m_Base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& xi = m_ParticlePositions[i];
				float& density = m_ParticleDensities[i];
				density = m_Volume * CubicKernelAVX::WZero();

				Scalar8 density_avx(0.0f);
				Scalar3f8 xi_avx(xi);

				forall_fluid_neighbors_avx(
					const Scalar8 Vj_avx = ConvertZero(m_Volume, count);
					density_avx += Vj_avx * CubicKernelAVX::W(xi_avx - xj_avx);
				);

				density += density_avx.Reduce();
				forall_volume_maps(
					density += Vj * PrecomputedCubicKernel::W(xi - xj);
				);

				density *= density0;
			}
		}
	}

	void DFSPHSimulation::ComputeDFSPHFactor()
	{
		auto* m_Base = this;
		const int numParticles = (int)m_ParticleCount;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				const glm::vec3& xi = m_ParticlePositions[i];

				float sum_grad_p_k;
				glm::vec3 grad_p_i;
				Scalar3f8 xi_avx(xi);
				Scalar8 sum_grad_p_k_avx(0.0f);
				Scalar3f8 grad_p_i_avx;
				grad_p_i_avx.SetZero();

				forall_fluid_neighbors_avx_nox(
					compute_Vj_gradW();
				const Scalar3f8 & gradC_j = V_gradW;
				sum_grad_p_k_avx += gradC_j.SquaredNorm();
				grad_p_i_avx = grad_p_i_avx + gradC_j;
				);

				sum_grad_p_k = sum_grad_p_k_avx.Reduce();
				grad_p_i[0] = grad_p_i_avx.x().Reduce();
				grad_p_i[1] = grad_p_i_avx.y().Reduce();
				grad_p_i[2] = grad_p_i_avx.z().Reduce();

				forall_volume_maps(
					const glm::vec3 grad_p_j = -Vj * PrecomputedCubicKernel::GradientW(xi - xj);
				grad_p_i -= grad_p_j;
				);

				sum_grad_p_k += glm::dot(grad_p_i, grad_p_i);

				float& factor = m_Factor[i];
				if (sum_grad_p_k > EPS) {
					factor = -static_cast<float>(1.0) / (sum_grad_p_k);
				}
				else {
					factor = 0.0;
				}
			}
		}
	}

	void DFSPHSimulation::DivergenceSolve()
	{
		const float h = m_TimeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		const unsigned int maxIter = m_MaxVolumeSolverIterations;
		const float maxError = m_MaxVolumeError;

		WarmStartDivergenceSolve();

		const int numParticles = (int)m_ParticleCount;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityChange(i, h);
				m_Factor[i] *= invH;
			}
		}

		m_VolumeSolverIterations = 0;

		float avg_density_err = 0.0;
		bool chk = false;

		while ((!chk || (m_VolumeSolverIterations < 1)) && (m_VolumeSolverIterations < maxIter))
		{
			chk = true;
			const float density0 = m_Density0;

			avg_density_err = 0.0;
			DivergenceSolveIteration(avg_density_err);

			// Maximal allowed density fluctuation
			// use maximal density error divided by time step size
			const float eta = (static_cast<float>(1.0) / h) * maxError * static_cast<float>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);

			m_VolumeSolverIterations++;
		}

		for (int i = 0; i < numParticles; i++)
			m_KappaVolume[i] *= h;


		for (int i = 0; i < numParticles; i++)
		{
			m_Factor[i] *= h;
		}
	}

	void DFSPHSimulation::WarmStartDivergenceSolve()
	{
		const float h = m_TimeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		const float density0 = m_Density0;
		const int numParticles = (int)m_ParticleCount;
		if (numParticles == 0)
			return;

		const Scalar8 invH_avx(invH);
		const Scalar8 h_avx(h);
		auto* m_Base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityChange(i, h);
				if (m_DensityAdvection[i] > 0.0)
					m_KappaVolume[i] = static_cast<float>(0.5) * std::max(m_KappaVolume[i], static_cast<float>(-0.5)) * invH;
				else
					m_KappaVolume[i] = 0.0;
			}

#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				//if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 0.0)
				{
					const float ki = m_KappaVolume[i];
					const glm::vec3& xi = m_ParticlePositions[i];
					glm::vec3& vi = m_ParticleVelocities[i];

					Scalar8 ki_avx(ki);
					Scalar3f8 xi_avx(xi);
					Scalar3f8 delta_vi;
					delta_vi.SetZero();

					forall_fluid_neighbors_avx_nox(
						compute_Vj_gradW();
						const Scalar8 densityFrac_avx(m_Density0 / density0);
						const Scalar8 kj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_KappaVolume[0], count);
						const Scalar8 kSum_avx = ki_avx + densityFrac_avx * kj_avx;

						delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));
					)

						vi[0] += delta_vi.x().Reduce();
					vi[1] += delta_vi.y().Reduce();
					vi[2] += delta_vi.z().Reduce();

					if (fabs(ki) > EPS)
					{
						forall_volume_maps(
							const glm::vec3 velChange = h * ki * Vj * PrecomputedCubicKernel::GradientW(xi - xj);
							vi += velChange;
							// bm_neighbor->addForce(xj, -model->getMass(i) * velChange * invH);
						);
					}
				}
			}
		}
	}

	void DFSPHSimulation::ComputeDensityChange(const unsigned int i, const float h)
	{
		const glm::vec3& xi = m_ParticlePositions[i];
		const glm::vec3& vi = m_ParticleVelocities[i];
		unsigned int numNeighbors = 0;

		Scalar8 densityAdv_avx(0.0f);
		const Scalar3f8 xi_avx(xi);
		Scalar3f8 vi_avx(vi);

		auto* m_Base = this;

		forall_fluid_neighbors_avx_nox(
			compute_Vj_gradW();
		const Scalar3f8 vj_avx = ConvertScalarZero(&GetNeighborList(0, 0, i)[j], &m_ParticleVelocities[0], count);
		densityAdv_avx += (vi_avx - vj_avx).Dot(V_gradW);
		);

		float& densityAdv = m_DensityAdvection[i];
		densityAdv = densityAdv_avx.Reduce();

		forall_volume_maps(
			glm::vec3 vj(0.0, 0.0, 0.0);
			densityAdv += Vj * glm::dot(vi - vj, PrecomputedCubicKernel::GradientW(xi - xj));
		);

		densityAdv = std::max(densityAdv, static_cast<float>(0.0));
		for (unsigned int pid = 0; pid < m_NeighborhoodSearch->GetPointSetCount(); pid++)
			numNeighbors += NumberOfNeighbors(0, pid, i);

		if (numNeighbors < 20) {
			densityAdv = 0.0;
		}
	}

	void DFSPHSimulation::DivergenceSolveIteration(float& avg_density_err)
	{
		const float density0 = m_Density0;
		const int numParticles = (int)m_ParticleCount;
		if (numParticles == 0)
			return;

		const float h = m_TimeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		float density_error = 0.0;
		const Scalar8 invH_avx(invH);
		const Scalar8 h_avx(h);

		auto* m_Base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				const float b_i =m_DensityAdvection[i];
				const float ki = b_i * m_Factor[i];

				m_KappaVolume[i] += ki;

				glm::vec3& vi = m_ParticleVelocities[i];
				const glm::vec3& xi = m_ParticlePositions[i];

				Scalar8 ki_avx(ki);
				Scalar3f8 xi_avx(xi);
				Scalar3f8 delta_vi;
				delta_vi.SetZero();

				forall_fluid_neighbors_avx_nox(
					compute_Vj_gradW();
					const Scalar8 densityFrac_avx(m_Density0 / density0);
					const Scalar8 densityAdvj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_DensityAdvection[0], count);
					const Scalar8 factorj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_Factor[0], count);

					const Scalar8 kj_avx = densityAdvj_avx * factorj_avx;
					const Scalar8 kSum_avx = MultiplyAndAdd(densityFrac_avx, kj_avx, ki_avx);

					delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));
				);

				vi[0] += delta_vi.x().Reduce();
				vi[1] += delta_vi.y().Reduce();
				vi[2] += delta_vi.z().Reduce();

				if (fabs(ki) > EPS)
				{
					forall_volume_maps(
						const glm::vec3 velChange = h * ki * Vj * PrecomputedCubicKernel::GradientW(xi - xj);
					vi += velChange;
					// bm_neighbor->addForce(xj, -model->getMass(i) * velChange * invH);
					);
				}
			}

#pragma omp for reduction(+:density_error) schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				ComputeDensityChange(i, h);
				density_error += m_DensityAdvection[i];
			}
		}

		avg_density_err = density0 * density_error / numParticles;
	}

	void DFSPHSimulation::ClearAccelerations()
	{
		const unsigned int count = m_ParticleCount;
		for (unsigned int i = 0; i < count; i++)
		{
			// Clear accelerations of dynamic particles
			if (m_ParticleMasses[i] != 0.0)
			{
				glm::vec3& a = m_ParticleAccelerations[i];
				a = m_Gravity;
			}
		}
	}

	void DFSPHSimulation::ComputeNonPressureForces()
	{
		// Surface tension
		//m_surfaceTension->OnUpdate();
		//m_viscosity->OnUpdate();
	}

	void DFSPHSimulation::UpdateTimeStepSize()
	{
		const float radius = m_ParticleRadius;
		float h = m_TimeStepSize;

		// Approximate max. position change due to current velocities
		float maxVel = 0.1;
		const float diameter = static_cast<float>(2.0) * radius;

		// fluid particles
		for (unsigned int i = 0; i < m_ParticleCount; i++)
		{
			const glm::vec3& vel = m_ParticleVelocities[i];
			const glm::vec3& accel = m_ParticleAccelerations[i];
			const float velMag = glm::length2(vel + accel * h);
			if (velMag > maxVel) {
				maxVel = velMag;
			}
		}

		// Approximate max. time step size 		
		h = m_CFLFactor * static_cast<float>(0.4) * (diameter / (sqrt(maxVel)));

		h = std::min(h, m_CFLMaxTimeStepSize);
		h = std::max(h, m_CFLMinTimeStepSize);

		m_TimeStepSize = h;
	}

	void DFSPHSimulation::PressureSolve()
	{
		const float h = m_TimeStepSize;
		const float h2 = h * h;
		const float invH = static_cast<float>(1.0) / h;
		const float invH2 = static_cast<float>(1.0) / h2;

		WarmStartPressureSolve();

		const int numParticles = (int)m_ParticleCount;
		const float density0 = m_Density0;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityAdv(i, numParticles, h, density0);
				m_Factor[i] *= invH2;
			}
		}

		m_PressureSolverIterations = 0;

		float avg_density_err = 0.0;
		bool chk = false;


		while ((!chk || (m_PressureSolverIterations < m_MinPressureSolverIteratations)) && (m_PressureSolverIterations < m_MaxPressureSolverIterations)) {
			chk = true;

			avg_density_err = 0.0;
			PressureSolveIteration(avg_density_err);

			// Maximal allowed density fluctuation
			const float eta = m_MaxPressureSolverError * static_cast<float>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);

			m_PressureSolverIterations++;
		}

		for (int i = 0; i < numParticles; i++)
			m_Kappa[i] *= h2;
	}

	void DFSPHSimulation::WarmStartPressureSolve()
	{
		const float h = m_TimeStepSize;
		const float h2 = h * h;
		const float invH = static_cast<float>(1.0) / h;
		const float invH2 = static_cast<float>(1.0) / h2;
		const float density0 = m_Density0;
		const int numParticles = (int)m_ParticleCount;

		auto* m_Base = this;

		const Scalar8 h_avx(h);
		if (numParticles == 0)
			return;

		const Scalar8 invH_avx(invH);

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				//m_simulationData.getKappa(fluidModelIndex, i) = max(m_simulationData.getKappa(fluidModelIndex, i)*invH2, -static_cast<float>(0.5) * density0*density0);
				ComputeDensityAdv(i, numParticles, h, density0);
				if (m_DensityAdvection[i] > 1.0)
					m_Kappa[i] = static_cast<float>(0.5) * std::max(m_Kappa[i], static_cast<float>(-0.00025)) * invH2;
				else
					m_Kappa[i] = 0.0;
			}

#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				const float ki = m_Kappa[i];
				const glm::vec3& xi = m_ParticlePositions[i];
				glm::vec3& vi = m_ParticleVelocities[i];

				Scalar8 ki_avx(ki);
				Scalar3f8 xi_avx(xi);
				Scalar3f8 delta_vi;
				delta_vi.SetZero();

				forall_fluid_neighbors_avx_nox(
					compute_Vj_gradW();
					const Scalar8 densityFrac_avx(m_Density0 / density0);
					const Scalar8 kj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_Kappa[0], count);
					const Scalar8 kSum_avx = ki_avx + densityFrac_avx * kj_avx;

					delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));			// ki, kj already contain inverse density	
				);

				vi[0] += delta_vi.x().Reduce();
				vi[1] += delta_vi.y().Reduce();
				vi[2] += delta_vi.z().Reduce();

				if (fabs(ki) > EPS)
				{
					forall_volume_maps(
						const glm::vec3 velChange = h * ki * Vj * PrecomputedCubicKernel::GradientW(xi - xj);
					vi += velChange;
					// bm_neighbor->addForce(xj, -model->getMass(i) * velChange * invH);
					);
				}
			}
		}
	}

	void DFSPHSimulation::ComputeDensityAdv(const unsigned int i, const int numParticles, const float h, const float density0)
	{
		const float& density = m_ParticleDensities[i];
		float& densityAdv = m_DensityAdvection[i];
		const glm::vec3& xi = m_ParticlePositions[i];
		const glm::vec3& vi = m_ParticleVelocities[i];
		float delta = 0.0;

		Scalar8 delta_avx(0.0f);
		const Scalar3f8 xi_avx(xi);
		Scalar3f8 vi_avx(vi);

		auto* m_Base = this;

		forall_fluid_neighbors_avx_nox(
			compute_Vj_gradW();
		const Scalar3f8 vj_avx = ConvertScalarZero(&GetNeighborList(0, 0, i)[j], &m_ParticleVelocities[0], count);
		delta_avx += (vi_avx - vj_avx).Dot(V_gradW);
		);

		delta = delta_avx.Reduce();

		forall_volume_maps(
			glm::vec3 vj(0, 0, 0);
		delta += Vj * glm::dot(vi - vj, PrecomputedCubicKernel::GradientW(xi - xj));
		);


		densityAdv = density / density0 + h * delta;
		densityAdv = std::max(densityAdv, static_cast<float>(1.0));
	}

	void DFSPHSimulation::PressureSolveIteration(float& avg_density_err)
	{
		const float density0 = m_Density0;
		const int numParticles = (int)m_ParticleCount;
		if (numParticles == 0)
			return;

		const float h = m_TimeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		float density_error = 0.0;
		const Scalar8 invH_avx(invH);
		const Scalar8 h_avx(h);
		auto* m_Base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < numParticles; i++) {
				const float b_i = m_DensityAdvection[i] - static_cast<float>(1.0);
				const float ki = b_i * m_Factor[i];

				m_Kappa[i] += ki;

				glm::vec3& vi = m_ParticleVelocities[i];
				const glm::vec3& xi = m_ParticlePositions[i];

				Scalar8 ki_avx(ki);
				Scalar3f8 xi_avx(xi);
				Scalar3f8 delta_vi;
				delta_vi.SetZero();

				forall_fluid_neighbors_avx_nox(
					compute_Vj_gradW();
					const Scalar8 densityFrac_avx(m_Density0 / density0);
					const Scalar8 densityAdvj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_DensityAdvection[0], count);
					const Scalar8 factorj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_Factor[0], count);

					const Scalar8 kj_avx = MultiplyAndSubtract(densityAdvj_avx, factorj_avx, factorj_avx);
					const Scalar8 kSum_avx = MultiplyAndAdd(densityFrac_avx, kj_avx, ki_avx);

					delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));
				);

				vi[0] += delta_vi.x().Reduce();
				vi[1] += delta_vi.y().Reduce();
				vi[2] += delta_vi.z().Reduce();

				if (fabs(ki) > EPS)
				{
					forall_volume_maps(
						const glm::vec3 velChange = h * ki * Vj * PrecomputedCubicKernel::GradientW(xi - xj);
					vi += velChange;
					);
				}
			}

#pragma omp for reduction(+:density_error) schedule(static) 
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityAdv(i, numParticles, h, density0);
				density_error += m_DensityAdvection[i] - static_cast<float>(1.0);
			}
		}
		avg_density_err = density0 * density_error / numParticles;
	}

	void DFSPHSimulation::PrecomputeValues()
	{
		m_PrecalculatedIndices.clear();
		m_PrecalculatedIndicesSamePhase.clear();
		m_PrecalculatedVolumeGradientW.clear();
		const int numParticles = (int)m_ParticleCount;

		auto& precomputed_indices = m_PrecalculatedIndices;
		auto& precomputed_indices_same_phase = m_PrecalculatedIndicesSamePhase;
		auto& precomputed_V_gradW = m_PrecalculatedVolumeGradientW;
		precomputed_indices.reserve(numParticles);
		precomputed_indices.push_back(0);

		precomputed_indices_same_phase.reserve(numParticles);

		unsigned int sumNeighborParticles = 0;
		unsigned int sumNeighborParticlesSamePhase = 0;
		for (int i = 0; i < numParticles; i++)
		{
			const unsigned int maxN = NumberOfNeighbors(0, 0, i);

			precomputed_indices_same_phase.push_back(sumNeighborParticles);

			// steps of 8 values due to avx
			sumNeighborParticles += maxN / 8;
			if (maxN % 8 != 0) {
				sumNeighborParticles++;
			}

			precomputed_indices.push_back(sumNeighborParticles);
		}

		if (sumNeighborParticles > precomputed_V_gradW.capacity()) {
			precomputed_V_gradW.reserve(static_cast<int>(1.5 * sumNeighborParticles));
		}
		precomputed_V_gradW.resize(sumNeighborParticles);

		auto* m_Base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& xi = m_ParticlePositions[i];
				const Scalar3f8 xi_avx(xi);
				const unsigned int base = precomputed_indices[i];
				unsigned int idx = 0;

				forall_fluid_neighbors_avx(
					const Scalar8 Vj_avx = ConvertZero(m_Volume, count);
				precomputed_V_gradW[base + idx] = CubicKernelAVX::GradientW(xi_avx - xj_avx) * Vj_avx;
				idx++;
				);
			}
		}
	}

	void DFSPHSimulation::InitFluidData()
	{
		m_Density0 = static_cast<float>(1000.0);
		float diam = static_cast<float>(2.0) * m_ParticleRadius;
		m_Volume = static_cast<float>(0.8) * diam * diam * diam;

		 // EdgeMesh mesh("Resources/Models/Cube.obj", { .6,  .6, .6 });

		{
			int c = 20;
			for (int x = -c / 2; x < c / 2; x++)
			{
				for (int y = -30 / 2; y < c / 2; y++)
				{
					for (int z = -c / 2; z < c / 2; z++)
					{
						m_ParticlePositions.push_back({ glm::vec3{x * diam, y * diam, z * diam} + glm::vec3{0.0, 5.0, 0.0} });
						m_ParticlePositions0.push_back({ glm::vec3{x * diam, y * diam, z * diam} + glm::vec3{0.0,5.0, 0.0} });
						m_ParticleVelocities.push_back({ 0.0, 0.0, 0.0 });
						m_ParticleVelocities0.push_back({ 0.0, 0, 0.0 });

						m_ParticleAccelerations.push_back({ 0.0, 0.0, 0.0 });
						m_ParticleDensities.push_back(m_Density0);
						m_ParticleMasses.push_back(m_Volume * m_Density0);
					}
				}
			}
		}

	/*	{
			glm::vec3 pos(0.0, 5.0, 0.0);
			float c = 500;
			for (int x = -c / 2; x < c / 2; x++)
			{
				for (int y = -c / 2; y < c / 2; y++)
				{
					for (int z = -c / 2; z < c / 2; z++)
					{
						glm::vec3 p = glm::vec3(x * diam, y * diam, z * diam) + pos;
						if (glm::distance(pos, p) <= 1.0) {

							glm::vec3 vel(0, -0,0);
							m_x.push_back(p);
							m_x0.push_back(p);
							m_v.push_back(vel);
							m_v0.push_back(vel);

							m_a.push_back({ 0.0, 0.0, 0.0 });
							m_density.push_back(m_density0);
							m_masses.push_back(m_V * m_density0);
						}
					}
				}
			}
		}*/


		/*for (const glm::vec3& sample : ParticleSampler::SampleMeshVolume(mesh, particleRadius, {20, 20, 20}, false, SampleMode::MediumDensity))
		{
			m_x.push_back({sample + glm::vec3{0, 3, 0}});
			m_v.push_back({ 0, 0, 0 });

			m_x0.push_back(m_x.back());
			m_v0.push_back(m_v.back());
			m_a.push_back({ 0, 0, 0 });
			m_density.push_back(0);
			m_masses.push_back(m_V * m_density0);
		}*/

		// Add fluid model TODO
		m_ParticleCount = m_ParticlePositions.size();

		m_NeighborhoodSearch->AddPointSet(&m_ParticlePositions[0][0], m_ParticleCount, true, true, true, this);
	}

	ViscosityWeiler2018::ViscosityWeiler2018(DFSPHSimulation* base)
	{
		m_maxIter = 100;
		m_MaxPressureSolverError = static_cast<float>(0.0001);
		m_boundaryViscosity =  1.0;
		m_ViscositySolver = 1;
		m_tangentialDistanceFactor = static_cast<float>(0.5);

		m_PressureSolverIterations = 0;
		m_vDiff.resize(base->GetParticleCount(), glm::vec3(0.0, 0.0, 0.0));
		m_Base = base;
	}

	void ViscosityWeiler2018::DiagonalMatrixElement(const unsigned int i, glm::mat3x3& result, void* userData, DFSPHSimulation* m_Base)
	{
		ViscosityWeiler2018* visco = (ViscosityWeiler2018*)userData;
		auto* sim = m_Base;

		const float density0 = sim->GetDensity0();
		const float d = 10.0;

		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float mu = visco->m_ViscositySolver * density0;
		const float mub = visco->m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;

		const float density_i = sim->GetParticleDensity(i);

		result[0][0] = 0.0;
		result[1][0] = 0.0;
		result[2][0] = 0.0;

		result[0][1] = 0.0;
		result[1][1] = 0.0;
		result[2][1] = 0.0;

		result[0][2] = 0.0;
		result[1][2] = 0.0;
		result[2][2] = 0.0;

		const glm::vec3& xi = sim->GetParticlePosition(i);

		const Scalar8 d_mu(d * mu);
		const Scalar8 d_mub(d * mub);
		const Scalar8 h2_001(0.01f * h2);
		const Scalar8 density0_avx(density0);
		const Scalar3f8 xi_avx(xi);
		const Scalar8 density_i_avx(density_i);

		Matrix3f8 res_avx;
		res_avx.SetZero();

		forall_fluid_neighbors_in_same_phase_avx(
			const Scalar8 density_j_avx = ConvertOne(&sim->GetNeighborList(0, 0, i)[j], &sim->GetParticleDensity(0), count);
			const Scalar3f8 xixj = xi_avx - xj_avx;
			const Scalar3f8 gradW = CubicKernelAVX::GradientW(xixj);
			const Scalar8 mj_avx = ConvertZero(sim->GetParticleMass(0), count);// all particles have the same mass TODO
			Matrix3f8 gradW_xij;
			DyadicProduct(gradW, xixj, gradW_xij);
			res_avx += gradW_xij * (d_mu * (mj_avx / density_j_avx) / (xixj.SquaredNorm() + h2_001));
		);

		if (mub != 0.0)
		{
			const auto& m_RigidBodies = m_Base->GetRigidBodies();

			forall_volume_maps(
				const glm::vec3 xixj = xi - xj;
				glm::vec3 normal = -xixj;
				const float nl = std::sqrt(glm::dot(normal, normal));

				if (nl > static_cast<float>(0.0001))
				{
					normal /= nl;

					glm::vec3 t1;
					glm::vec3 t2;
					GetOrthogonalVectors(normal, t1, t2);

					const float dist = visco->m_tangentialDistanceFactor * h;
					const glm::vec3 x1 = xj - t1 * dist;
					const glm::vec3 x2 = xj + t1 * dist;
					const glm::vec3 x3 = xj - t2 * dist;
					const glm::vec3 x4 = xj + t2 * dist;

					const glm::vec3 xix1 = xi - x1;
					const glm::vec3 xix2 = xi - x2;
					const glm::vec3 xix3 = xi - x3;
					const glm::vec3 xix4 = xi - x4;

					const glm::vec3 gradW1 = PrecomputedCubicKernel::GradientW(xix1);
					const glm::vec3 gradW2 = PrecomputedCubicKernel::GradientW(xix2);
					const glm::vec3 gradW3 = PrecomputedCubicKernel::GradientW(xix3);
					const glm::vec3 gradW4 = PrecomputedCubicKernel::GradientW(xix4);

					const float vol = static_cast<float>(0.25) * Vj;

					result += (d * mub * vol / (glm::dot(xix1, xix1) + 0.01f * h2)) * glm::outerProduct(xix1, gradW1);
					result += (d * mub * vol / (glm::dot(xix2, xix2) + 0.01f * h2)) * glm::outerProduct(xix2, gradW2);
					result += (d * mub * vol / (glm::dot(xix3, xix3) + 0.01f * h2)) * glm::outerProduct(xix3, gradW3);
					result += (d * mub * vol / (glm::dot(xix4, xix4) + 0.01f * h2)) * glm::outerProduct(xix4, gradW4);
				}
			);
		}

		result += res_avx.Reduce();
		result = glm::identity<glm::mat3x3>() - (dt / density_i) * result;
	}

	void ViscosityWeiler2018::ComputeRHS(std::vector<float>& b, std::vector<float>& g)
	{
		const int numParticles = (int)m_Base->GetParticleCount();
		auto* sim = m_Base;
		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float density0 = sim->GetDensity0();
		const float mu = m_ViscositySolver * density0;
		const float mub = m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;
		float d = 10.0;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) nowait
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& vi = sim->GetParticleVelocity(i);
				const glm::vec3& xi = sim->GetParticlePosition(i);
				const float density_i = sim->GetParticleDensity(i);
				const float m_i = sim->GetParticleMass(i);
				glm::vec3 bi(0.0, 0.0, 0.0);

				if (mub != 0.0)
				{
					const auto& m_RigidBodies = m_Base->GetRigidBodies();

					forall_volume_maps(
						const glm::vec3 xixj = xi - xj;
						glm::vec3 normal = -xixj;
						const float nl = std::sqrt(glm::dot(normal, normal));

						if (nl > static_cast<float>(0.0001)) {
							normal /= nl;

							glm::vec3 t1;
							glm::vec3 t2;
							GetOrthogonalVectors(normal, t1, t2);

							const float dist = m_tangentialDistanceFactor * h;
							const glm::vec3 x1 = xj - t1 * dist;
							const glm::vec3 x2 = xj + t1 * dist;
							const glm::vec3 x3 = xj - t2 * dist;
							const glm::vec3 x4 = xj + t2 * dist;

							const glm::vec3 xix1 = xi - x1;
							const glm::vec3 xix2 = xi - x2;
							const glm::vec3 xix3 = xi - x3;
							const glm::vec3 xix4 = xi - x4;

							const glm::vec3 gradW1 = PrecomputedCubicKernel::GradientW(xix1);
							const glm::vec3 gradW2 = PrecomputedCubicKernel::GradientW(xix2);
							const glm::vec3 gradW3 = PrecomputedCubicKernel::GradientW(xix3);
							const glm::vec3 gradW4 = PrecomputedCubicKernel::GradientW(xix4);

							const float vol = static_cast<float>(0.25) * Vj;

							glm::vec3 v1(0.0, 0.0, 0.0);
							glm::vec3 v2(0.0, 0.0, 0.0);
							glm::vec3 v3(0.0, 0.0, 0.0);
							glm::vec3 v4(0.0, 0.0, 0.0);

							const glm::vec3 a1 = d * mub * vol * glm::dot(v1, xix1) / (glm::dot(xix1, xix1) + 0.01f * h2) * gradW1;
							const glm::vec3 a2 = d * mub * vol * glm::dot(v2, xix2) / (glm::dot(xix2, xix2) + 0.01f * h2) * gradW2;
							const glm::vec3 a3 = d * mub * vol * glm::dot(v3, xix3) / (glm::dot(xix3, xix3) + 0.01f * h2) * gradW3;
							const glm::vec3 a4 = d * mub * vol * glm::dot(v4, xix4) / (glm::dot(xix4, xix4) + 0.01f * h2) * gradW4;
							bi += a1 + a2 + a3 + a4;
						}
					);
				}

				b[3 * i + 0] = vi[0] - dt / density_i * bi[0];
				b[3 * i + 1] = vi[1] - dt / density_i * bi[1];
				b[3 * i + 2] = vi[2] - dt / density_i * bi[2];

				g[3 * i + 0] = vi[0] + m_vDiff[i][0];
				g[3 * i + 1] = vi[1] + m_vDiff[i][1];
				g[3 * i + 2] = vi[2] + m_vDiff[i][2];
			}
		}
	}

	void ViscosityWeiler2018::ApplyForces(const std::vector<float>& x)
	{
		const int numParticles = (int)m_Base->GetParticleCount();
		auto* sim = m_Base;
		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float density0 = sim->GetDensity0();
		const float mu = m_ViscositySolver * density0;
		const float mub = m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;
		float d = 10.0;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)
			for (int i = 0; i < (int)numParticles; i++)
			{
				glm::vec3& ai = sim->GetParticleAcceleration(i);
				const glm::vec3 newVi(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
				ai += (1.0f / dt) * (newVi - sim->GetParticleVelocity(i));
				m_vDiff[i] = (newVi - sim->GetParticleVelocity(i));

				const glm::vec3& xi = sim->GetParticlePosition(i);
				const float density_i = sim->GetParticleDensity(i);
				const float m_i = sim->GetParticleMass(i);

				if (mub != 0.0)
				{
					const auto& m_RigidBodies = m_Base->GetRigidBodies();
					forall_volume_maps(
						const glm::vec3 xixj = xi - xj;
					glm::vec3 normal = -xixj;
					const float nl = std::sqrt(glm::dot(normal, normal));

					if (nl > static_cast<float>(0.0001)) {
						normal /= nl;

						glm::vec3 t1;
						glm::vec3 t2;
						GetOrthogonalVectors(normal, t1, t2);

						const float dist = m_tangentialDistanceFactor * h;
						const glm::vec3 x1 = xj - t1 * dist;
						const glm::vec3 x2 = xj + t1 * dist;
						const glm::vec3 x3 = xj - t2 * dist;
						const glm::vec3 x4 = xj + t2 * dist;

						const glm::vec3 xix1 = xi - x1;
						const glm::vec3 xix2 = xi - x2;
						const glm::vec3 xix3 = xi - x3;
						const glm::vec3 xix4 = xi - x4;

						const glm::vec3 gradW1 = PrecomputedCubicKernel::GradientW(xix1);
						const glm::vec3 gradW2 = PrecomputedCubicKernel::GradientW(xix2);
						const glm::vec3 gradW3 = PrecomputedCubicKernel::GradientW(xix3);
						const glm::vec3 gradW4 = PrecomputedCubicKernel::GradientW(xix4);

						const float vol = static_cast<float>(0.25) * Vj;

						glm::vec3 v1(0.0, 0.0, 0.0);
						glm::vec3 v2(0.0, 0.0, 0.0);
						glm::vec3 v3(0.0, 0.0, 0.0);
						glm::vec3 v4(0.0, 0.0, 0.0);

						const glm::vec3 a1 = d * mub * vol * glm::dot(newVi - v1, xix1) / (glm::dot(xix1, xix1) + 0.01f * h2) * gradW1;
						const glm::vec3 a2 = d * mub * vol * glm::dot(newVi - v2, xix2) / (glm::dot(xix2, xix2) + 0.01f * h2) * gradW2;
						const glm::vec3 a3 = d * mub * vol * glm::dot(newVi - v3, xix3) / (glm::dot(xix3, xix3) + 0.01f * h2) * gradW3;
						const glm::vec3 a4 = d * mub * vol * glm::dot(newVi - v4, xix4) / (glm::dot(xix4, xix4) + 0.01f * h2) * gradW4;
					}
					);
				}
			}
		}
	}

	void ViscosityWeiler2018::OnUpdate() {
		const unsigned int numParticles = (int)m_Base->GetParticleCount();
		if (numParticles == 0) {
			return;
		}

		const float density0 = m_Base->GetDensity0();
		const float h = m_Base->GetTimeStepSize();

		MatrixReplacement A(3 * numParticles, MatrixVecProd, (void*)this, m_Base);
		m_solver.GetPreconditioner().Init(numParticles, DiagonalMatrixElement, (void*)this, m_Base);

		m_solver.m_tolerance = m_MaxPressureSolverError;
		m_solver.m_MaxPressureSolverIterations = m_maxIter;
		m_solver.Compute(A);

		std::vector<float> b(3 * numParticles);
		std::vector<float> g(3 * numParticles);
		std::vector<float> x(3 * numParticles);

		ComputeRHS(b, g);
		m_solver.SolveWithGuess(b, g, x);
		ApplyForces(x);
	}

	void ViscosityWeiler2018::MatrixVecProd(const std::vector<float>& vec, std::vector<float>& result, void* userData, DFSPHSimulation* m_Base)
	{
		ViscosityWeiler2018* visco = (ViscosityWeiler2018*)userData;
		auto* sim = m_Base;
		const unsigned int numParticles = sim->GetParticleCount();

		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float density0 = sim->GetDensity0();
		const float mu = visco->m_ViscositySolver * density0;
		const float mub = visco->m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;
		const float d = 10.0;

		const Scalar8 d_mu_rho0(d * mu * density0);
		const Scalar8 d_mub(d * mub);
		const Scalar8 h2_001(0.01f * h2);
		const Scalar8 density0_avx(density0);

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& xi = sim->GetParticlePosition(i);
				glm::vec3 ai;
				ai = glm::vec3(0.0, 0.0, 0.0);
				const float density_i = sim->GetParticleDensity(0);
				const glm::vec3& vi = glm::vec3(vec[i * 3 + 0], vec[i * 3 + 1], vec[i * 3 + 2]);

				const Scalar3f8 xi_avx(xi);
				const Scalar3f8 vi_avx(vi);
				const Scalar8 density_i_avx(density_i);
				const Scalar8 mi_avx(sim->GetParticleMass(i));

				Scalar3f8 delta_ai_avx;
				delta_ai_avx.SetZero();

				forall_fluid_neighbors_in_same_phase_avx(
					compute_Vj_gradW_samephase();
					const Scalar8 density_j_avx = ConvertOne(&sim->GetNeighborList(0, 0, i)[j], &sim->GetParticleDensity(0), count);
					const Scalar3f8 xixj = xi_avx - xj_avx;
					const Scalar3f8 vj_avx = ConvertScalarZero(&sim->GetNeighborList(0, 0, i)[j], &vec[0], count);

					delta_ai_avx = delta_ai_avx + (V_gradW * ((d_mu_rho0 / density_j_avx) * (vi_avx - vj_avx).Dot(xixj) / (xixj.SquaredNorm() + h2_001)));
				);

				if (mub != 0.0)
				{
					const auto& m_RigidBodies = m_Base->GetRigidBodies();

					forall_volume_maps(
						const glm::vec3 xixj = xi - xj;
						glm::vec3 normal = -xixj;
						const float nl = std::sqrt(glm::dot(normal, normal));
						if (nl > static_cast<float>(0.0001)) {
							normal /= nl;

							glm::vec3 t1;
							glm::vec3 t2;
							GetOrthogonalVectors(normal, t1, t2);

							const float dist = visco->m_tangentialDistanceFactor * h;
							const glm::vec3 x1 = xj - t1 * dist;
							const glm::vec3 x2 = xj + t1 * dist;
							const glm::vec3 x3 = xj - t2 * dist;
							const glm::vec3 x4 = xj + t2 * dist;

							const glm::vec3 xix1 = xi - x1;
							const glm::vec3 xix2 = xi - x2;
							const glm::vec3 xix3 = xi - x3;
							const glm::vec3 xix4 = xi - x4;

							const glm::vec3 gradW1 = PrecomputedCubicKernel::GradientW(xix1);
							const glm::vec3 gradW2 = PrecomputedCubicKernel::GradientW(xix2);
							const glm::vec3 gradW3 = PrecomputedCubicKernel::GradientW(xix3);
							const glm::vec3 gradW4 = PrecomputedCubicKernel::GradientW(xix4);

							const float vol = static_cast<float>(0.25) * Vj;
							glm::vec3 v1(0.0, 0.0, 0.0);
							glm::vec3 v2(0.0, 0.0, 0.0);
							glm::vec3 v3(0.0, 0.0, 0.0);
							glm::vec3 v4(0.0, 0.0, 0.0);

							const glm::vec3 a1 = (d * mub * vol * glm::dot(vi, xix1) / (glm::dot(xix1, xix1) + 0.01f * h2)) * gradW1;
							const glm::vec3 a2 = (d * mub * vol * glm::dot(vi, xix2) / (glm::dot(xix2, xix2) + 0.01f * h2)) * gradW2;
							const glm::vec3 a3 = (d * mub * vol * glm::dot(vi, xix3) / (glm::dot(xix3, xix3) + 0.01f * h2)) * gradW3;
							const glm::vec3 a4 = (d * mub * vol * glm::dot(vi, xix4) / (glm::dot(xix4, xix4) + 0.01f * h2)) * gradW4;
							ai += a1 + a2 + a3 + a4;
						}
					);
				}

				ai[0] += delta_ai_avx.x().Reduce();
				ai[1] += delta_ai_avx.y().Reduce();
				ai[2] += delta_ai_avx.z().Reduce();

				result[3 * i] = vec[3 * i] - dt / density_i * ai[0];
				result[3 * i + 1] = vec[3 * i + 1] - dt / density_i * ai[1];
				result[3 * i + 2] = vec[3 * i + 2] - dt / density_i * ai[2];
			}
		}
	}

	SurfaceTensionZorillaRitter2020::SurfaceTensionZorillaRitter2020(DFSPHSimulation* base)
		:
		m_SurfaceTensionSolver(.3f)
		, m_Csd(10000) // 10000 // 36000 // 48000 // 60000
		, m_tau(0.5)
		, m_r2mult(0.8f)
		, m_r1(base->GetParticleSupportRadius())
		, m_r2(m_r2mult* m_r1)
		, m_class_k(74.688796680497925f)
		, m_class_d(28)
		, m_temporal_smoothing(false)
		, m_CsdFix(-1)
		, m_class_d_off(2)
		, m_pca_N_mix(0.75f)
		, m_pca_C_mix(0.5f)
		, m_neighs_limit(16)
		, m_CS_smooth_passes(1)
	{
		m_Base = base;
		m_mc_normals          .resize(base->GetParticleCount(), { 0.0f, 0.0f, 0.0f });
		m_mc_normals_smooth   .resize(base->GetParticleCount(), { 0.0f, 0.0f, 0.0f });
		m_pca_normals         .resize(base->GetParticleCount(), { 0.0f, 0.0f, 0.0f });
		m_final_curvatures    .resize(base->GetParticleCount(),   0.0f);
		m_pca_curv            .resize(base->GetParticleCount(),   0.0f);
		m_pca_curv_smooth     .resize(base->GetParticleCount(),   0.0f);
		m_mc_curv             .resize(base->GetParticleCount(),   0.0f);
		m_mc_curv_smooth      .resize(base->GetParticleCount(),   0.0f);
		m_final_curvatures_old.resize(base->GetParticleCount(),   0.0f);
		m_classifier_input    .resize(base->GetParticleCount(),   0.0f);
		m_classifier_output   .resize(base->GetParticleCount(),   0.0f);
	}

	void SurfaceTensionZorillaRitter2020::OnUpdate()
	{
		float timeStep = m_Base->GetTimeStepSize();

		m_r2 = m_r1 * m_r2mult;

		auto* sim = m_Base;

		const float supportRadius = sim->GetParticleSupportRadius();
		const unsigned int numParticles = sim->GetParticleCount();
		const float k = m_SurfaceTensionSolver;

		unsigned int NrOfSamples;

		if (m_CsdFix > 0)
			NrOfSamples = m_CsdFix;
		else
			NrOfSamples = int(m_Csd * timeStep);

		// ################################################################################################
		// ## first pass, compute classification and first estimation for normal and curvature (Montecarlo)
		// ################################################################################################

		int erased = 0;
#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				// init or reset arrays
				m_mc_normals[i] = glm::vec3(0,  0, 0);
				m_pca_normals[i] = glm::vec3(0,  0, 0);
				m_mc_normals_smooth[i] = glm::vec3(0,  0, 0);

				m_mc_curv[i] = 0.0;
				m_mc_curv_smooth[i] = 0.0;
				m_pca_curv[i] = 0.0;
				m_pca_curv_smooth[i] = 0.0;
				m_final_curvatures[i] = 0.0;


				// -- compute center of mass of current particle

				glm::vec3 centerofMasses = glm::vec3(0,  0, 0);
				int numberOfNeighbours = sim->NumberOfNeighbors(0, 0, i);

				if (numberOfNeighbours == 0)
				{
					m_mc_curv[i] = static_cast<float>(1.0) / supportRadius;
					continue;
				}

				const glm::vec3& xi = sim->GetParticlePosition(i);

				forall_fluid_neighbors_in_same_phase(
					glm::vec3 xjxi = (xj - xi);
					centerofMasses += xjxi;
				);

				centerofMasses /= supportRadius;

				// cache classifier input, could also be recomputed later to avoid caching
				m_classifier_input[i] = glm::length(centerofMasses) / static_cast<float>(numberOfNeighbours);


				// -- if it is a surface classified particle
				if (ClassifyParticleConfigurable(m_classifier_input[i], numberOfNeighbours)) //EvaluateNetwork also possible
				{

					// -- create monte carlo samples on particle
					std::vector<glm::vec3> points = GetSphereSamplesLookUp(
						NrOfSamples, supportRadius, i * NrOfSamples, haltonVec323, static_cast<int>(haltonVec323.size())); // 8.5 // 15.0(double) // 9.0(float)

					//  -- remove samples covered by neighbor spheres
					forall_fluid_neighbors_in_same_phase(
						glm::vec3 xjxi = (xj - xi);
						for (int p = static_cast<int>(points.size()) - 1; p >= 0; --p)
						{
							glm::vec3 vec = (points[p] - xjxi);
							float dist = glm::length2(vec);

							if (dist <= pow((m_r2 / m_r1), 2) * supportRadius * supportRadius) {
								points.erase(points.begin() + p);
								erased++;
							}
						})

						// -- estimate normal by left over sample directions
						for (int p = static_cast<int>(points.size()) - 1; p >= 0; --p)
							m_mc_normals[i] += points[p];


					// -- if surface classified and non-overlapping neighborhood spheres
					if (points.size() > 0)
					{
						m_mc_normals[i] = glm::normalize(m_mc_normals[i]);

						// -- estimate curvature by sample ratio and particle radii
						m_mc_curv[i] = (static_cast<float>(1.0) / supportRadius) * static_cast<float>(-2.0) * pow((static_cast<float>(1.0) - (m_r2 * m_r2 / (m_r1 * m_r1))), static_cast<float>(-0.5)) *
							cos(acos(static_cast<float>(1.0) - static_cast<float>(2.0) * (static_cast<float>(points.size()) / static_cast<float>(NrOfSamples))) + asin(m_r2 / m_r1));

						m_classifier_output[i] = 1.0; // -- used to visualize surface points (blue in the paper)
					}
					else
					{
						// -- correct false positives to inner points
						m_mc_normals[i] = glm::vec3(0,  0, 0);
						m_mc_curv[i] = 0.0;
						m_classifier_output[i] = 0.5; // -- used for visualize post-correction points (white in the paper)
					}
				}
				else
				{
					// -- used to visualize inner points (green in the paper)
					m_classifier_output[i] = 0.0;
				}

			}
		}

		// ################################################################################################
		// ## second pass, compute normals and curvature and compute PCA normal 
		// ################################################################################################

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				if (m_mc_normals[i] != glm::vec3(0,  0, 0))
				{
					const glm::vec3& xi = sim->GetParticlePosition(i);
					glm::vec3 normalCorrection = glm::vec3(0,  0, 0);
					glm::vec3& ai = sim->GetParticleAcceleration(i);


					float correctionForCurvature = 0;
					float correctionFactor = 0.0;

					glm::vec3 centroid = xi;
					glm::vec3 surfCentDir = glm::vec3(0,  0, 0);

					// collect neighbors
					std::multimap<float, size_t> neighs;

					glm::mat3x3 t = glm::mat3x3();
					int t_count = 0;
					glm::vec3 neighCent = glm::vec3(0,  0, 0);

					int nrNeighhbors = sim->NumberOfNeighbors(0, 0, i);

					forall_fluid_neighbors_in_same_phase(
						if (m_mc_normals[neighborIndex] != glm::vec3(0,  0, 0))
						{
							glm::vec3& xj = sim->GetParticlePosition(neighborIndex);
							glm::vec3 xjxi = (xj - xi);

							surfCentDir += xjxi;
							centroid += xj;
							t_count++;

							float distanceji = glm::length(xjxi);

							normalCorrection += m_mc_normals[neighborIndex] * (1 - distanceji / supportRadius);
							correctionForCurvature += m_mc_curv[neighborIndex] * (1 - distanceji / supportRadius);
							correctionFactor += (1 - distanceji / supportRadius);
						}
						)

						normalCorrection = glm::normalize(normalCorrection);
						m_mc_normals_smooth[i] = (1 - m_tau) * m_mc_normals[i] + m_tau * normalCorrection;
						m_mc_normals_smooth[i] = glm::normalize(m_mc_normals_smooth[i]);

						m_mc_curv_smooth[i] =
							((static_cast<float>(1.0) - m_tau) * m_mc_curv[i] + m_tau * correctionForCurvature) /
							((static_cast<float>(1.0) - m_tau) + m_tau * correctionFactor);
				}
			}
		}


		// ################################################################################################
		// ## third pass, final blending and temporal smoothing
		// ################################################################################################

		m_CS_smooth_passes = std::max(1, m_CS_smooth_passes);

		for (int si = 0; si < m_CS_smooth_passes; si++)
		{
			// smoothing pass 2 for sphericity
#pragma omp parallel default(shared)
			{
#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					if (m_mc_normals[i] != glm::vec3(0, 0, 0))
					{
						int count = 0;
						float CsCorr = 0.0;

						const glm::vec3& xi = sim->GetParticlePosition(i);

						forall_fluid_neighbors_in_same_phase(
							if (m_mc_normals[neighborIndex] != glm::vec3(0, 0, 0))
							{
								CsCorr += m_pca_curv[neighborIndex];
								count++;
							})


							if (count > 0)
								m_pca_curv_smooth[i] = static_cast<float>(0.25) * m_pca_curv_smooth[i] + static_cast<float>(0.75) * CsCorr / static_cast<float>(count);
							else
								m_pca_curv_smooth[i] = m_pca_curv[i];

							m_pca_curv_smooth[i] /= supportRadius;
							m_pca_curv_smooth[i] *= 20.0;

							if (m_pca_curv_smooth[i] > 0.0)
								m_pca_curv_smooth[i] = std::min(0.5f / supportRadius, m_pca_curv_smooth[i]);
							else
								m_pca_curv_smooth[i] = std::max(-0.5f / supportRadius, m_pca_curv_smooth[i]);


							glm::vec3 final_normal = glm::vec3(0, 0, 0);
							float     final_curvature = m_mc_curv_smooth[i];

							final_normal = m_mc_normals_smooth[i];
							final_curvature = m_mc_curv_smooth[i];

							if (m_temporal_smoothing)
								m_final_curvatures[i] = static_cast<float>(0.05) * final_curvature + static_cast<float>(0.95) * m_final_curvatures_old[i];
							else
								m_final_curvatures[i] = final_curvature;

							glm::vec3 force = final_normal * k * m_final_curvatures[i];

							glm::vec3& ai = sim->GetParticleAcceleration(i);
							ai -= (1 / sim->GetParticleMass(i)) * force;

							m_final_curvatures_old[i] = m_final_curvatures[i];
					}
					else // non surface particle blend 0.0 curvature
					{

						if (m_temporal_smoothing)
							m_final_curvatures[i] = static_cast<float>(0.95) * m_final_curvatures_old[i];
						else
							m_final_curvatures[i] = 0.0;

						m_final_curvatures_old[i] = m_final_curvatures[i];
					}
				}
			}
		}
	}

	bool SurfaceTensionZorillaRitter2020::ClassifyParticleConfigurable(double com, int non, double d_offset)
	{
		double neighborsOnTheLine = m_class_k * com + m_class_d + d_offset;

		if (non <= neighborsOnTheLine) {
			return true;
		}
		else {
			return false;
		}
	}

	std::vector<glm::vec3> SurfaceTensionZorillaRitter2020::GetSphereSamplesLookUp(int N, float supportRadius, int start, const std::vector<float>& vec3, int mod)
	{
		std::vector<glm::vec3> points(N);
		int s = (start / 3) * 3;

		for (int i = 0; i < N; i++)
		{
			int i3 = s + 3 * i;
			points[i] = supportRadius * glm::vec3(vec3[i3 % mod], vec3[(i3 + 1) % mod], vec3[(i3 + 2) % mod]);
		}

		return points;
	}
}
