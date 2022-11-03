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
		const glm::vec3 &xj = m_x[neighborIndex]; \
		code \
	} \

#define forall_volume_maps(code) \
	const float Vj = m_boundaryModels->GetBoundaryVolume(i);  \
	if (Vj > 0.0) \
	{ \
		const glm::vec3 &xj = m_boundaryModels->GetBoundaryXj(i); \
		code \
	} \

#define forall_fluid_neighbors_avx(code)\
	const unsigned int maxN = m_base->NumberOfNeighbors(0, 0, i); \
	for (unsigned int j = 0; j < maxN; j += 8) \
	{ \
		const unsigned int count = std::min(maxN - j, 8u); \
		const Scalar3f8 xj_avx = ConvertScalarZero(&m_base->GetNeighborList(0, 0, i)[j], &m_x[0], count); \
		code \
	} \

#define forall_fluid_neighbors_in_same_phase(code) \
	for (unsigned int j = 0; j < m_base->NumberOfNeighbors(0, 0, i); j++) \
	{ \
		const unsigned int neighborIndex = m_base->GetNeighbor(0, 0, i, j); \
		const glm::vec3 &xj = m_base->m_x[neighborIndex]; \
		code \
	} 

#define forall_fluid_neighbors_avx_nox(code) \
	unsigned int idx = 0; \
	const unsigned int maxN = m_base->NumberOfNeighbors(0, 0, i); \
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
		const Scalar3f8 xj_avx = ConvertScalarZero(&sim->GetNeighborList(0, 0, i)[j], &sim->m_x[0], count); \
		code \
	} \

#define compute_Vj_gradW() const Scalar3f8& V_gradW = m_precomp_V_gradW[m_precompIndices[i] + idx];
#define compute_Vj_gradW_samephase() const Scalar3f8& V_gradW = sim->m_precomp_V_gradW[sim->m_precompIndicesSamePhase[i] + j / 8];
namespace fe {
	DFSPHSimulation::DFSPHSimulation(const DFSPHSimulationDescription& desc)
	{
		m_boundarySimulator = nullptr;
		m_isStaticScene = true;
		m_numberOfStepsPerRenderUpdate = 4;
		m_renderWalls = 4;
		m_doPause = true;
		m_pauseAt = -1.0;
		m_stopAt = -1.0;
		m_useParticleCaching = true;
		m_useGUI = true;
		m_enableRigidBodyVTKExport = false;
		m_enableRigidBodyExport = false;
		m_enableStateExport = false;
		m_enableAsyncExport = false;
		m_enableObjectSplitting = false;
		m_framesPerSecond = 25;
		m_framesPerSecondState = 1;
		m_nextFrameTime = 0.0;
		m_nextFrameTimeState = 0.0;
		m_frameCounter = 1;
		m_isFirstFrame = true;
		m_isFirstFrameVTK = true;
		m_firstState = true;
		m_colorField.resize(1, "velocity");
		m_colorMapType.resize(1, 0);
		m_renderMinValue.resize(1, 0.0);
		m_renderMaxValue.resize(1, 5.0);
		m_particleAttributes = "velocity";
		m_timeStepCB = nullptr;
		m_resetCB = nullptr;
		m_updateGUI = false;
		m_currentObjectId = 0;

		m_Material = Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl"));
		m_Material->Set("color", { 0.4, 0.4, 0.4 });


		// Init the scene 
		timeStepSize = 0.001;
		particleRadius = 0.025;
		// Rigid bodies
		{
			BoundaryData* data = new BoundaryData();
			data->meshFile = "Resources/Models/Cube.obj";
			data->translation = { 0, -0.25, 0 };
			data->rotation = glm::angleAxis(glm::radians(45.f), glm::vec3(1.f, 0.f, 0.f));
			data->scale = { 5, 0.5, 5 };
			data->dynamic = false;
			data->isWall = false;
			data->samplingMode = 0;
			data->isAnimated = false;
			data->mapInvert = false;
			data->mapThickness = 0.0;
			data->mapResolution = { 30, 20, 30 };

			boundaryModels.push_back(data);
		}

		// Init sim 
		{
			m_enableZSort = false;
			m_gravitation = { 0.0, -9.81, 0.0 };
			// m_gravitation = { 0.0, 0.0, 0.0 };
			m_cflFactor = 1;
			m_cflMethod = 1; // standard
			m_cflMinTimeStepSize = 0.0001;
			m_cflMaxTimeStepSize = 0.005;
			m_boundaryHandlingMethod = 2; // Bender et al. 2019

			// REG FORCES
			SetParticleRadius(particleRadius);

			m_neighborhoodSearch = new NeighborhoodSearch(m_supportRadius, false);
			m_neighborhoodSearch->SetRadius(m_supportRadius);
		}

		m_W_zero = PrecomputedCubicKernel::WZero();
		BuildModel();

		// Init boundary sim 
		m_boundarySimulator = new StaticBoundarySimulator(this);

		// FluidModel
		// update scalar field 
		//m_scalarField.resize(1);
		//m_scalarField[0].resize(m_numActiveParticles);
		//for (size_t i = 0; i < m_numActiveParticles; i++)
		//{

		//}
		DefferedInit();

		m_surfaceTension = new SurfaceTensionZorillaRitter2020(this);
		m_viscosity = new ViscosityWeiler2018(this);
	}

	DFSPHSimulation::~DFSPHSimulation()
	{
	}

	void DFSPHSimulation::OnUpdate()
	{
		if (paused) {
			return;
		}

		const float h = timeStepSize;

		{
			if (m_counter % 500 == 0) {
				// Simulation::getCurrent()->performNeighborhoodSearchSort();
				{
					m_neighborhoodSearch->ZSort();

					const unsigned int numPart = m_numActiveParticles;
					if (numPart > 0) {
						auto const& d = m_neighborhoodSearch->GetPointSet(m_pointSetIndex);

						d.SortField(&m_x[0]);
						d.SortField(&m_v[0]);
						d.SortField(&m_a[0]);
						d.SortField(&m_masses[0]);
						d.SortField(&m_density[0]);

						// Viscosity
						d.SortField(&m_viscosity->m_vDiff[0]);

						// Surface tension
						d.SortField(&m_surfaceTension->m_mc_normals[0]);
						d.SortField(&m_surfaceTension->m_final_curvatures[0]);
						d.SortField(&m_surfaceTension->m_pca_curv[0]);
						d.SortField(&m_surfaceTension->m_pca_curv_smooth[0]);
						d.SortField(&m_surfaceTension->m_mc_curv[0]);
						d.SortField(&m_surfaceTension->m_mc_curv_smooth[0]);
						d.SortField(&m_surfaceTension->m_mc_normals_smooth[0]);
						d.SortField(&m_surfaceTension->m_pca_normals[0]);
						d.SortField(&m_surfaceTension->m_final_curvatures_old[0]);
						d.SortField(&m_surfaceTension->m_classifier_input[0]);
						d.SortField(&m_surfaceTension->m_classifier_output[0]);
					}
				}
			
				// m_simulationData.performNeighborhoodSearchSort();
				m_simulationData.PerformNeighborhoodSearchSort(this);
			}

			m_counter++;
			m_neighborhoodSearch->FindNeighbors();
		}

		PrecomputeValues();

		ComputeVolumeAndBoundaryX();

		ComputeDensities();

		ComputeDFSPHFactor();

		DivergenceSolve();

		ClearAccelerations();

		// Non-Pressure forces
		m_surfaceTension->OnUpdate();
		m_viscosity->OnUpdate();

		UpdateTimeStepSize();

		{
			const unsigned int numParticles = m_numActiveParticles;
#pragma omp parallel default(shared)
			{
#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++) {
					glm::vec3& vel = m_v[i];
					vel += h * m_a[i];
				}
			}
		}

		PressureSolve();

		{
			const unsigned int numParticles = m_numActiveParticles;
#pragma omp parallel default(shared)
			{
#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					glm::vec3& xi = m_x[i];
					const glm::vec3& vi = m_v[i];
					xi += h * vi;
				}
			}
		}
	}

	void DFSPHSimulation::OnRenderTemp()
	{
		glm::vec3 offset = { 0, 4, 0 };
		auto* r = m_boundaryModels[0].GetRigidBody();
		glm::mat4 t = glm::translate(glm::mat4(1.0f), offset + glm::vec3{ 0, -0.25, 0 });
		t = glm::scale(t, { .52, .45, .52 });
		t = glm::rotate(t, glm::radians(49.0f), { 1, 0, 0 });

		m_Material->Set("model", t);

		Renderer::DrawTriangles(r->GetGeometry().GetVAO(), r->GetGeometry().GetVertexCount(), m_Material);

		for (size_t i = 0; i < m_numParticles; i++)
		{
			// default
			Renderer::DrawPoint(m_x[i] + offset, { .5, .3, 1, 1 }, particleRadius * 35);

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
	}

	void DFSPHSimulation::InitVolumeMap(std::vector<glm::vec3>& x, std::vector<glm::ivec3>& faces, const BoundaryData* boundaryData, const bool md5, const bool isDynamic, BoundaryModelBender2019* boundaryModel)
	{
		SDF* volumeMap;
		glm::ivec3 resolutionSDF = boundaryData->mapResolution;
		const float supportRadius = m_supportRadius;

		std::string mapFile = "Resources/cache.cdm";
		volumeMap = new SDF(mapFile);
		boundaryModel->SetMap(volumeMap);
		return;

		{
			////////////////////////////////////////////////////////////////////////
			//	Generate distance field of object using Discregrid
			//////////////////////////////////////////////////////////////////////////

			std::vector<glm::vec3> doubleVec;
			doubleVec.resize(x.size());
			for (unsigned int i = 0; i < x.size(); i++) {
				doubleVec[i] = { x[i].x, x[i].y , x[i].z };
			}
			EdgeMesh sdfMesh(x, faces);

			MeshDistance md(sdfMesh);
			BoundingBox domain;
			for (auto const& x_ : x)
			{
				domain.Extend(x_);
			}

			const float tolerance = boundaryData->mapThickness;
			domain.max += (8.0f * supportRadius + tolerance) * glm::vec3(1, 1, 1);
			domain.min -= (8.0f * supportRadius + tolerance) * glm::vec3(1, 1, 1);

			std::cout << "Domain - min: " << domain.min[0] << ", " << domain.min[1] << ", " << domain.min[2] << std::endl;
			std::cout << "Domain - max: " << domain.max[0] << ", " << domain.max[1] << ", " << domain.max[2] << std::endl;

			volumeMap = new SDF(domain, resolutionSDF);
			auto func = SDF::ContinuousFunction{};

			float sign = 1.0;
			if (boundaryData->mapInvert)
				sign = -1.0;

			const float particleRadius = particleRadius;
			func = [&md, &sign, &tolerance, &particleRadius](glm::vec3 const& xi) {return sign * (md.SignedDistanceCached(xi) - tolerance); };

			LOG("GENERATE SDF");
			volumeMap->AddFunction(func);

			//////////////////////////////////////////////////////////////////////////
			// Generate volume map of object
			//////////////////////////////////////////////////////////////////////////

			BoundingBox int_domain;
			int_domain.min = glm::vec3(-supportRadius);
			int_domain.max = glm::vec3(supportRadius);

			float factor = 1.0;

			auto volume_func = [&](glm::vec3 const& x)
			{
				float dist_x = volumeMap->Interpolate(0u, x);

				if (dist_x > (1.0 + 1.0 /*/ factor*/) * supportRadius)
				{
					return 0.0f;
				}

				auto integrand = [&](glm::vec3 const& xi) -> float
				{
					if (glm::dot(xi, xi) > supportRadius * supportRadius)
						return 0.0;

					auto dist = volumeMap->Interpolate(0u, x + xi);

					if (dist <= 0.0f)
						return 1.0f;// -0.001 * dist / supportRadius;
					if (dist < 1.0f / factor * supportRadius)
						return static_cast<float>(CubicKernel::W(factor * static_cast<float>(dist)) / CubicKernel::WZero());
					return 0.0f;
				};

				float res = 0.0f;
				res = 0.8f * GaussQuadrature::Integrate(integrand, int_domain, 30);

				return res;
			};

			LOG("GENERATE VOLUME MAP");
			const bool no_reduction = true;

			auto predicate_function = [&](glm::vec3 const& x_)
			{
				if (no_reduction)
				{
					return true;
				}
				auto x = glm::max(x_, (glm::vec3)glm::min(volumeMap->GetDomain().min, volumeMap->GetDomain().max));
				auto dist = volumeMap->Interpolate(0u, x);
				if (dist == std::numeric_limits<float>::max())
				{
					return false;
				}

				return fabs(dist) < 4.0 * supportRadius;
			};

			volumeMap->AddFunction(volume_func, predicate_function);

			boundaryModel->SetMap(volumeMap);
		}
	}

	void DFSPHSimulation::UpdateVMVelocity()
	{
		//for (size_t i = 0; i < boundaryModels.size(); i++)
		//{
		//	BoundaryModelBender2019* bm = m_boundaryModels;
		//	StaticRigidBody* rbo = bm->GetRigidBody();

		//}
	}

	void DFSPHSimulation::SetParticleRadius(float val)
	{
		m_supportRadius = static_cast<float>(4.0) * val;
		PrecomputedCubicKernel::SetRadius(m_supportRadius);
		CubicKernelAVX::SetRadius(m_supportRadius);
	}

	void DFSPHSimulation::BuildModel()
	{
		InitFluidData();

		// Init timestep
		m_simulationData.Init(this);
	}

	void DFSPHSimulation::ComputeVolumeAndBoundaryX()
	{
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < m_numParticles; i++)
			{
				const glm::vec3& xi = m_x[i];
				ComputeVolumeAndBoundaryX(i, xi);
			}
		}
	}

	void DFSPHSimulation::ComputeVolumeAndBoundaryX(const unsigned int i, const glm::vec3& xi)
	{
		BoundaryModelBender2019* bm = m_boundaryModels;
		glm::vec3& boundaryXj = bm->GetBoundaryXj(i);
		boundaryXj = { 0.0, 0.0, 0.0 };
		float& boundaryVolume = bm->GetBoundaryVolume(i);
		boundaryVolume = 0.0;

		const glm::vec3& t = { 0, -0.25, 0 };
		glm::mat3 R = glm::toMat3(bm->GetRigidBody()->m_q);

		/*R[0][0] = 1;
		R[0][1] = 0;
		R[0][2] = 0;
		R[1][0] = 0;
		R[1][1] = 0.707107;
		R[1][2] = -0.707107;
		R[2][0] = 0;
		R[2][1] = 0.707107;
		R[2][2] = 0.707107;*/

		glm::dvec3 normal;
		const glm::dvec3 localXi = (glm::transpose(R) * ((glm::dvec3)xi - (glm::dvec3)t));

		std::array<unsigned int, 32> cell;
		glm::dvec3 c0;
		std::array<double, 32> N;
		std::array<std::array<double, 3>, 32> dN;
		bool chk = bm->m_map->DetermineShapeFunctions(0, localXi, cell, c0, N, &dN);

		double dist = std::numeric_limits<double>::max();
		if (chk) {
			dist = bm->m_map->Interpolate(0, localXi, cell, c0, N, &normal, &dN);
		}

		bool animateParticle = false;

		if ((dist > 0.0) && (static_cast<float>(dist) < m_supportRadius))
		{
			const double volume = bm->m_map->Interpolate(1, localXi, cell, c0, N);
			if ((volume > 0.0) && (volume != std::numeric_limits<double>::max()))
			{
				boundaryVolume = static_cast<float>(volume);

				normal = R * normal;
				const double nl = std::sqrt(glm::dot(normal, normal));

				if (nl > 1.0e-9)
				{
					normal /= nl;
					const float d = std::max((static_cast<float>(dist) + static_cast<float>(0.5) * particleRadius), static_cast<float>(2.0) * particleRadius);
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
					float delta = static_cast<float>(2.0) * particleRadius - static_cast<float>(dist);
					delta = std::min(delta, static_cast<float>(0.1) * particleRadius);		// get up in small steps
					m_x[i] = (xi + delta * (glm::vec3)normal);
					// adapt velocity in normal direction
					// m_v[i] = 1.0 / timeStepSize * delta * normal;
					m_v[i] = { 0.0, 0.0, 0.0 };
				}
			}
			boundaryVolume = 0.0;
		}
	}

	void DFSPHSimulation::ComputeDensities()
	{
		const float density0 = m_density0;
		const unsigned int numParticles = m_numActiveParticles;
		auto* m_base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& xi = m_x[i];
				float& density = m_density[i];
				density = m_V * CubicKernelAVX::WZero();

				Scalar8 density_avx(0.0f);
				Scalar3f8 xi_avx(xi);

				forall_fluid_neighbors_avx(
					const Scalar8 Vj_avx = ConvertZero(m_V, count);
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
		auto* m_base = this;
		const int numParticles = (int)m_numActiveParticles;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				const glm::vec3& xi = m_x[i];

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

				float& factor = m_simulationData.GetFactor(0, i);
				if (sum_grad_p_k > m_eps) {
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
		const float h = timeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		const unsigned int maxIter = m_maxIterationsV;
		const float maxError = m_maxErrorV;

		WarmStartDivergenceSolve();

		const int numParticles = (int)m_numActiveParticles;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityChange(i, h);
				m_simulationData.GetFactor(0, i) *= invH;
			}
		}

		m_iterationsV = 0;

		float avg_density_err = 0.0;
		bool chk = false;

		while ((!chk || (m_iterationsV < 1)) && (m_iterationsV < maxIter))
		{
			chk = true;
			const float density0 = m_density0;

			avg_density_err = 0.0;
			DivergenceSolveIteration(avg_density_err);

			// Maximal allowed density fluctuation
			// use maximal density error divided by time step size
			const float eta = (static_cast<float>(1.0) / h) * maxError * static_cast<float>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);

			m_iterationsV++;
		}

		for (int i = 0; i < numParticles; i++)
			m_simulationData.GetKappaV(0, i) *= h;


		for (int i = 0; i < numParticles; i++)
		{
			m_simulationData.GetFactor(0, i) *= h;
		}
	}

	void DFSPHSimulation::WarmStartDivergenceSolve()
	{
		const float h = timeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		const float density0 = m_density0;
		const int numParticles = (int)m_numActiveParticles;
		if (numParticles == 0)
			return;

		const Scalar8 invH_avx(invH);
		const Scalar8 h_avx(h);
		auto* m_base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityChange(i, h);
				if (m_simulationData.GetDensityAdv(0, i) > 0.0)
					m_simulationData.GetKappaV(0, i) = static_cast<float>(0.5) * std::max(m_simulationData.GetKappaV(0, i), static_cast<float>(-0.5)) * invH;
				else
					m_simulationData.GetKappaV(0, i) = 0.0;
			}

#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				//if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 0.0)
				{
					const float ki = m_simulationData.GetKappaV(0, i);
					const glm::vec3& xi = m_x[i];
					glm::vec3& vi = m_v[i];

					Scalar8 ki_avx(ki);
					Scalar3f8 xi_avx(xi);
					Scalar3f8 delta_vi;
					delta_vi.SetZero();

					forall_fluid_neighbors_avx_nox(
						compute_Vj_gradW();
					const Scalar8 densityFrac_avx(m_density0 / density0);
					const Scalar8 kj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_simulationData.GetKappaV(0, 0), count);
					const Scalar8 kSum_avx = ki_avx + densityFrac_avx * kj_avx;

					delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));
					)

						vi[0] += delta_vi.x().Reduce();
					vi[1] += delta_vi.y().Reduce();
					vi[2] += delta_vi.z().Reduce();

					if (fabs(ki) > m_eps)
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
		const glm::vec3& xi = m_x[i];
		const glm::vec3& vi = m_v[i];
		unsigned int numNeighbors = 0;

		Scalar8 densityAdv_avx(0.0f);
		const Scalar3f8 xi_avx(xi);
		Scalar3f8 vi_avx(vi);

		auto* m_base = this;

		forall_fluid_neighbors_avx_nox(
			compute_Vj_gradW();
		const Scalar3f8 vj_avx = ConvertScalarZero(&GetNeighborList(0, 0, i)[j], &m_v[0], count);
		densityAdv_avx += (vi_avx - vj_avx).Dot(V_gradW);
		);

		float& densityAdv = m_simulationData.GetDensityAdv(0, i);
		densityAdv = densityAdv_avx.Reduce();

		forall_volume_maps(
			glm::vec3 vj(0.0, 0.0, 0.0);
		densityAdv += Vj * glm::dot(vi - vj, PrecomputedCubicKernel::GradientW(xi - xj));
		);

		densityAdv = std::max(densityAdv, static_cast<float>(0.0));
		for (unsigned int pid = 0; pid < m_neighborhoodSearch->GetPointSetCount(); pid++)
			numNeighbors += NumberOfNeighbors(0, pid, i);

		if (numNeighbors < 20) {
			densityAdv = 0.0;
		}
	}

	void DFSPHSimulation::DivergenceSolveIteration(float& avg_density_err)
	{
		const float density0 = m_density0;
		const int numParticles = (int)m_numActiveParticles;
		if (numParticles == 0)
			return;

		const float h = timeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		float density_error = 0.0;
		const Scalar8 invH_avx(invH);
		const Scalar8 h_avx(h);

		auto* m_base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				const float b_i = m_simulationData.GetDensityAdv(0, i);
				const float ki = b_i * m_simulationData.GetFactor(0, i);

				m_simulationData.GetKappaV(0, i) += ki;

				glm::vec3& vi = m_v[i];
				const glm::vec3& xi = m_x[i];

				Scalar8 ki_avx(ki);
				Scalar3f8 xi_avx(xi);
				Scalar3f8 delta_vi;
				delta_vi.SetZero();

				forall_fluid_neighbors_avx_nox(
					compute_Vj_gradW();
				const Scalar8 densityFrac_avx(m_density0 / density0);
				const Scalar8 densityAdvj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_simulationData.GetDensityAdv(0, 0), count);
				const Scalar8 factorj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_simulationData.GetFactor(0, 0), count);

				const Scalar8 kj_avx = densityAdvj_avx * factorj_avx;
				const Scalar8 kSum_avx = MultiplyAndAdd(densityFrac_avx, kj_avx, ki_avx);

				delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));
				);

				vi[0] += delta_vi.x().Reduce();
				vi[1] += delta_vi.y().Reduce();
				vi[2] += delta_vi.z().Reduce();

				if (fabs(ki) > m_eps)
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
				density_error += m_simulationData.GetDensityAdv(0, i);
			}
		}

		avg_density_err = density0 * density_error / numParticles;
	}

	void DFSPHSimulation::ClearAccelerations()
	{
		const unsigned int count = m_numActiveParticles;
		for (unsigned int i = 0; i < count; i++)
		{
			// Clear accelerations of dynamic particles
			if (m_masses[i] != 0.0)
			{
				glm::vec3& a = m_a[i];
				a = m_gravitation;
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
		const float radius = particleRadius;
		float h = timeStepSize;

		// Approximate max. position change due to current velocities
		float maxVel = 0.1;
		const float diameter = static_cast<float>(2.0) * radius;

		// fluid particles
		for (unsigned int i = 0; i < m_numActiveParticles; i++)
		{
			const glm::vec3& vel = m_v[i];
			const glm::vec3& accel = m_a[i];
			const float velMag = glm::length2(vel + accel * h);
			if (velMag > maxVel) {
				maxVel = velMag;
			}
		}

		// Approximate max. time step size 		
		h = m_cflFactor * static_cast<float>(0.4) * (diameter / (sqrt(maxVel)));

		h = std::min(h, m_cflMaxTimeStepSize);
		h = std::max(h, m_cflMinTimeStepSize);

		timeStepSize = h;
	}

	void DFSPHSimulation::PressureSolve()
	{
		const float h = timeStepSize;
		const float h2 = h * h;
		const float invH = static_cast<float>(1.0) / h;
		const float invH2 = static_cast<float>(1.0) / h2;

		WarmStartPressureSolve();

		const int numParticles = (int)m_numActiveParticles;
		const float density0 = m_density0;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityAdv(i, numParticles, h, density0);
				m_simulationData.GetFactor(0, i) *= invH2;
			}
		}

		m_iterations = 0;

		float avg_density_err = 0.0;
		bool chk = false;


		while ((!chk || (m_iterations < m_minIterations)) && (m_iterations < m_maxIterations)) {
			chk = true;

			avg_density_err = 0.0;
			PressureSolveIteration(avg_density_err);

			// Maximal allowed density fluctuation
			const float eta = m_maxError * static_cast<float>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);

			m_iterations++;
		}

		for (int i = 0; i < numParticles; i++)
			m_simulationData.GetKappa(0, i) *= h2;
	}

	void DFSPHSimulation::WarmStartPressureSolve()
	{
		const float h = timeStepSize;
		const float h2 = h * h;
		const float invH = static_cast<float>(1.0) / h;
		const float invH2 = static_cast<float>(1.0) / h2;
		const float density0 = m_density0;
		const int numParticles = (int)m_numActiveParticles;

		auto* m_base = this;

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
				if (m_simulationData.GetDensityAdv(0, i) > 1.0)
					m_simulationData.GetKappa(0, i) = static_cast<float>(0.5) * std::max(m_simulationData.GetKappa(0, i), static_cast<float>(-0.00025)) * invH2;
				else
					m_simulationData.GetKappa(0, i) = 0.0;
			}

#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				const float ki = m_simulationData.GetKappa(0, i);
				const glm::vec3& xi = m_x[i];
				glm::vec3& vi = m_v[i];

				Scalar8 ki_avx(ki);
				Scalar3f8 xi_avx(xi);
				Scalar3f8 delta_vi;
				delta_vi.SetZero();

				forall_fluid_neighbors_avx_nox(
					compute_Vj_gradW();
				const Scalar8 densityFrac_avx(m_density0 / density0);
				const Scalar8 kj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_simulationData.GetKappa(0, 0), count);
				const Scalar8 kSum_avx = ki_avx + densityFrac_avx * kj_avx;

				delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));			// ki, kj already contain inverse density	
				);

				vi[0] += delta_vi.x().Reduce();
				vi[1] += delta_vi.y().Reduce();
				vi[2] += delta_vi.z().Reduce();

				if (fabs(ki) > m_eps)
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
		const float& density = m_density[i];
		float& densityAdv = m_simulationData.GetDensityAdv(0, i);
		const glm::vec3& xi = m_x[i];
		const glm::vec3& vi = m_v[i];
		float delta = 0.0;

		Scalar8 delta_avx(0.0f);
		const Scalar3f8 xi_avx(xi);
		Scalar3f8 vi_avx(vi);

		auto* m_base = this;

		forall_fluid_neighbors_avx_nox(
			compute_Vj_gradW();
		const Scalar3f8 vj_avx = ConvertScalarZero(&GetNeighborList(0, 0, i)[j], &m_v[0], count);
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
		const float density0 = m_density0;
		const int numParticles = (int)m_numActiveParticles;
		if (numParticles == 0)
			return;

		const float h = timeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		float density_error = 0.0;
		const Scalar8 invH_avx(invH);
		const Scalar8 h_avx(h);
		auto* m_base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < numParticles; i++) {
				const float b_i = m_simulationData.GetDensityAdv(0, i) - static_cast<float>(1.0);
				const float ki = b_i * m_simulationData.GetFactor(0, i);

				m_simulationData.GetKappa(0, i) += ki;

				glm::vec3& vi = m_v[i];
				const glm::vec3& xi = m_x[i];

				Scalar8 ki_avx(ki);
				Scalar3f8 xi_avx(xi);
				Scalar3f8 delta_vi;
				delta_vi.SetZero();

				forall_fluid_neighbors_avx_nox(
					compute_Vj_gradW();
				const Scalar8 densityFrac_avx(m_density0 / density0);
				const Scalar8 densityAdvj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_simulationData.GetDensityAdv(0, 0), count);
				const Scalar8 factorj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_simulationData.GetFactor(0, 0), count);

				const Scalar8 kj_avx = MultiplyAndSubtract(densityAdvj_avx, factorj_avx, factorj_avx);
				const Scalar8 kSum_avx = MultiplyAndAdd(densityFrac_avx, kj_avx, ki_avx);

				delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));
				);

				vi[0] += delta_vi.x().Reduce();
				vi[1] += delta_vi.y().Reduce();
				vi[2] += delta_vi.z().Reduce();

				if (fabs(ki) > m_eps)
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
				density_error += m_simulationData.GetDensityAdv(0, i) - static_cast<float>(1.0);
			}
		}
		avg_density_err = density0 * density_error / numParticles;
	}

	void DFSPHSimulation::PrecomputeValues()
	{
		m_precompIndices.clear();
		m_precompIndicesSamePhase.clear();
		m_precomp_V_gradW.clear();
		const int numParticles = (int)m_numActiveParticles;

		auto& precomputed_indices = m_precompIndices;
		auto& precomputed_indices_same_phase = m_precompIndicesSamePhase;
		auto& precomputed_V_gradW = m_precomp_V_gradW;
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

		auto* m_base = this;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& xi = m_x[i];
				const Scalar3f8 xi_avx(xi);
				const unsigned int base = precomputed_indices[i];
				unsigned int idx = 0;

				forall_fluid_neighbors_avx(
					const Scalar8 Vj_avx = ConvertZero(m_V, count);
				precomputed_V_gradW[base + idx] = CubicKernelAVX::GradientW(xi_avx - xj_avx) * Vj_avx;
				idx++;
				);
			}
		}
	}

	void DFSPHSimulation::InitFluidData()
	{
		m_density0 = static_cast<float>(1000.0);
		float diam = static_cast<float>(2.0) * particleRadius;
		m_V = static_cast<float>(0.8) * diam * diam * diam;

		// EdgeMesh mesh("Resources/Models/Cube.obj", { .6,  .6, .6 });

		{
			int c = 20;
			for (int x = -c / 2; x < c / 2; x++)
			{
				for (int y = -c / 2; y < c / 2; y++)
				{
					for (int z = -c / 2; z < c / 2; z++)
					{
						m_x.push_back({ glm::vec3{x * diam, y * diam, z * diam} + glm::vec3{0.0, 2.0, 0.0} });
						m_x0.push_back({ glm::vec3{x * diam, y * diam, z * diam} + glm::vec3{0.0, 2.0, 0.0} });
						m_v.push_back({ 0.0, 0.0, 0.0 });
						m_v0.push_back({ 0.0, 0.0, 0.0 });

						m_a.push_back({ 0.0, 0.0, 0.0 });
						m_density.push_back(0.0);
						m_masses.push_back(m_V * m_density0);
					}
				}
			}
		}

		/*{
			glm::vec3 pos(0.0, 2.0, 0.0);
			float c = 30;
			for (int x = -c / 2; x < c / 2; x++)
			{
				for (int y = -c / 2; y < c / 2; y++)
				{
					for (int z = -c / 2; z < c / 2; z++)
					{
						glm::vec3 p = glm::vec3(x * diam, y * diam, z * diam) + pos;
						if (glm::distance(pos, p) <= .5) {

							glm::vec3 vel(0, 10, 0);
							m_x.push_back(p);
							m_x0.push_back(p);
							m_v.push_back(vel);
							m_v0.push_back(vel);

							m_a.push_back({ 0.0, 0.0, 0.0 });
							m_density.push_back(0.0);
							m_masses.push_back(m_V * m_density0);
						}
					}
				}
			}
		}*/


		//for (const glm::vec3& sample : ParticleSampler::SampleMeshVolume(mesh, particleRadius, {20, 20, 20}, false, SampleMode::MaxDensity))
		//{
		//	m_x.push_back({sample + glm::vec3{0, 3, 0}});
		//	m_v.push_back({ 0, 0, 0 });

		//	m_x0.push_back(m_x.back());
		//	m_v0.push_back(m_v.back());
		//	m_a.push_back({ 0, 0, 0 });
		//	m_density.push_back(0);
		//	m_masses.push_back(m_V * m_density0);
		//}

		// Add fluid model TODO
		m_numParticles = m_x.size();

		m_pointSetIndex = m_neighborhoodSearch->AddPointSet(&m_x[0][0], m_numParticles, true, true, true, this);
		m_numActiveParticles0 = m_numParticles;
		m_numActiveParticles = m_numActiveParticles0;
	}

	void DFSPHSimulation::DefferedInit()
	{
		m_boundarySimulator->InitBoundaryData();
		m_simulationIsInitialized = true;

		// deffered init sim 
		// m_surfaceTension->deferredInit();
		m_boundarySimulator->DefferedInit();
	}

	SimulationDataDFSPH::SimulationDataDFSPH()
		: m_factor(),
		m_kappa(),
		m_kappaV(),
		m_density_adv()
	{}

	void SimulationDataDFSPH::Init(DFSPHSimulation* sim)
	{
		m_factor.resize(1);
		m_kappa.resize(1);
		m_kappaV.resize(1);
		m_density_adv.resize(1);

		for (unsigned int i = 0; i < 1; i++)
		{
			m_factor[i].resize(sim->m_numParticles, 0.0);
			m_kappa[i].resize(sim->m_numParticles, 0.0);
			m_kappaV[i].resize(sim->m_numParticles, 0.0);
			m_density_adv[i].resize(sim->m_numParticles, 0.0);
		}
	}

	void SimulationDataDFSPH::PerformNeighborhoodSearchSort(DFSPHSimulation* base)
	{
		const unsigned int numPart = base->m_numActiveParticles;
		if (numPart > 0)
		{
			auto const& d = base->m_neighborhoodSearch->GetPointSet(base->m_pointSetIndex);
			d.SortField(&m_kappa[0][0]);
			d.SortField(&m_kappaV[0][0]);
		}
	}

	ViscosityWeiler2018::ViscosityWeiler2018(DFSPHSimulation* base)
	{
		m_maxIter = 100;
		m_maxError = static_cast<float>(0.001);
		m_boundaryViscosity = 1.f;
		m_viscosity = 1.f;
		m_tangentialDistanceFactor = static_cast<float>(0.5);

		m_iterations = 0;
		m_vDiff.resize(base->m_numParticles, glm::vec3(0.0, 0.0, 0.0));
		m_base = base;
	}

	void ViscosityWeiler2018::DiagonalMatrixElement(const unsigned int i, glm::mat3x3& result, void* userData, DFSPHSimulation* m_base)
	{
		ViscosityWeiler2018* visco = (ViscosityWeiler2018*)userData;
		auto* sim = m_base;

		const float density0 = sim->m_density0;
		const float d = 10.0;

		const float h = sim->m_supportRadius;
		const float h2 = h * h;
		const float dt = sim->timeStepSize;
		const float mu = visco->m_viscosity * density0;
		const float mub = visco->m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;

		const float density_i = sim->m_density[i];

		result[0][0] = 0.0;
		result[1][0] = 0.0;
		result[2][0] = 0.0;

		result[0][1] = 0.0;
		result[1][1] = 0.0;
		result[2][1] = 0.0;

		result[0][2] = 0.0;
		result[1][2] = 0.0;
		result[2][2] = 0.0;

		const glm::vec3& xi = sim->m_x[i];

		const Scalar8 d_mu(d * mu);
		const Scalar8 d_mub(d * mub);
		const Scalar8 h2_001(0.01f * h2);
		const Scalar8 density0_avx(density0);
		const Scalar3f8 xi_avx(xi);
		const Scalar8 density_i_avx(density_i);

		Matrix3f8 res_avx;
		res_avx.SetZero();

		forall_fluid_neighbors_in_same_phase_avx(
			const Scalar8 density_j_avx = ConvertOne(&sim->GetNeighborList(0, 0, i)[j], &sim->m_density[0], count);
			const Scalar3f8 xixj = xi_avx - xj_avx;
			const Scalar3f8 gradW = CubicKernelAVX::GradientW(xixj);
			const Scalar8 mj_avx = ConvertZero(sim->m_masses[0], count);// all particles have the same mass TODO
			Matrix3f8 gradW_xij;
			DyadicProduct(gradW, xixj, gradW_xij);
			res_avx += gradW_xij * (d_mu * (mj_avx / density_j_avx) / (xixj.SquaredNorm() + h2_001));
		);

		if (mub != 0.0)
		{
			BoundaryModelBender2019* m_boundaryModels = sim->m_boundaryModels;

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
		const int numParticles = (int)m_base->m_numActiveParticles;
		auto* sim = m_base;
		const float h = sim->m_supportRadius;
		const float h2 = h * h;
		const float dt = sim->timeStepSize;
		const float density0 = sim->m_density0;
		const float mu = m_viscosity * density0;
		const float mub = m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;
		float d = 10.0;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) nowait
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& vi = sim->m_v[i];
				const glm::vec3& xi = sim->m_x[i];
				const float density_i = sim->m_density[i];
				const float m_i = sim->m_masses[i];
				glm::vec3 bi(0.0, 0.0, 0.0);

				if (mub != 0.0)
				{
					BoundaryModelBender2019* m_boundaryModels = sim->m_boundaryModels;

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
		const int numParticles = (int)m_base->m_numActiveParticles;
		auto* sim = m_base;
		const float h = sim->m_supportRadius;
		const float h2 = h * h;
		const float dt = sim->timeStepSize;
		const float density0 = sim->m_density0;
		const float mu = m_viscosity * density0;
		const float mub = m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;
		float d = 10.0;

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)
			for (int i = 0; i < (int)numParticles; i++)
			{
				glm::vec3& ai = sim->m_a[i];
				const glm::vec3 newVi(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
				ai += (1.0f / dt) * (newVi - sim->m_v[i]);
				m_vDiff[i] = (newVi - sim->m_v[i]);

				const glm::vec3& xi = sim->m_x[i];
				const float density_i = sim->m_density[i];
				const float m_i = sim->m_masses[i];

				if (mub != 0.0)
				{
					BoundaryModelBender2019* m_boundaryModels = sim->m_boundaryModels;

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
		const unsigned int numParticles = (int)m_base->m_numActiveParticles;
		if (numParticles == 0) {
			return;
		}

		const float density0 = m_base->m_density0;
		const float h = m_base->timeStepSize;

		MatrixReplacement A(3 * numParticles, MatrixVecProd, (void*)this, m_base);
		m_solver.GetPreconditioner().Init(numParticles, DiagonalMatrixElement, (void*)this, m_base);

		m_solver.m_tolerance = m_maxError;
		m_solver.m_maxIterations = m_maxIter;
		m_solver.Compute(A);

		std::vector<float> b(3 * numParticles);
		std::vector<float> g(3 * numParticles);
		std::vector<float> x(3 * numParticles);

		ComputeRHS(b, g);
		m_solver.SolveWithGuess(b, g, x);
		ApplyForces(x);
	}

	void ViscosityWeiler2018::MatrixVecProd(const std::vector<float>& vec, std::vector<float>& result, void* userData, DFSPHSimulation* m_base)
	{
		ViscosityWeiler2018* visco = (ViscosityWeiler2018*)userData;
		auto* sim = m_base;
		const unsigned int numParticles = sim->m_numActiveParticles;

		const float h = sim->m_supportRadius;
		const float h2 = h * h;
		const float dt = sim->timeStepSize;
		const float density0 = sim->m_density0;
		const float mu = visco->m_viscosity * density0;
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
				const glm::vec3& xi = sim->m_x[i];
				glm::vec3 ai;
				ai = glm::vec3(0.0, 0.0, 0.0);
				const float density_i = sim->m_density[i];
				const glm::vec3& vi = glm::vec3(vec[i * 3 + 0], vec[i * 3 + 1], vec[i * 3 + 2]);

				const Scalar3f8 xi_avx(xi);
				const Scalar3f8 vi_avx(vi);
				const Scalar8 density_i_avx(density_i);
				const Scalar8 mi_avx(sim->m_masses[i]);

				Scalar3f8 delta_ai_avx;
				delta_ai_avx.SetZero();

				forall_fluid_neighbors_in_same_phase_avx(
					compute_Vj_gradW_samephase();
				const Scalar8 density_j_avx = ConvertOne(&sim->GetNeighborList(0, 0, i)[j], &sim->m_density[0], count);
				const Scalar3f8 xixj = xi_avx - xj_avx;
				const Scalar3f8 vj_avx = ConvertScalarZero(&sim->GetNeighborList(0, 0, i)[j], &vec[0], count);

				delta_ai_avx = delta_ai_avx + (V_gradW * ((d_mu_rho0 / density_j_avx) * (vi_avx - vj_avx).Dot(xixj) / (xixj.SquaredNorm() + h2_001)));
				);

				if (mub != 0.0)
				{
					BoundaryModelBender2019* m_boundaryModels = sim->m_boundaryModels;

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
		m_surfaceTension(3.f)
		, m_Csd(10000) // 10000 // 36000 // 48000 // 60000
		, m_tau(0.5)
		, m_r2mult(0.8f)
		, m_r1(base->m_supportRadius)
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
		m_base = base;
		m_mc_normals          .resize(base->m_numParticles, { 0.0f, 0.0f, 0.0f });
		m_mc_normals_smooth   .resize(base->m_numParticles, { 0.0f, 0.0f, 0.0f });
		m_pca_normals         .resize(base->m_numParticles, { 0.0f, 0.0f, 0.0f });
		m_final_curvatures    .resize(base->m_numParticles,   0.0f);
		m_pca_curv            .resize(base->m_numParticles,   0.0f);
		m_pca_curv_smooth     .resize(base->m_numParticles,   0.0f);
		m_mc_curv             .resize(base->m_numParticles,   0.0f);
		m_mc_curv_smooth      .resize(base->m_numParticles,   0.0f);
		m_final_curvatures_old.resize(base->m_numParticles,   0.0f);
		m_classifier_input    .resize(base->m_numParticles,   0.0f);
		m_classifier_output   .resize(base->m_numParticles,   0.0f);
	}

	void SurfaceTensionZorillaRitter2020::OnUpdate()
	{
		float timeStep = m_base->timeStepSize;

		m_r2 = m_r1 * m_r2mult;

		auto* sim = m_base;

		const float supportRadius = sim->m_supportRadius;
		const unsigned int numParticles = sim->m_numActiveParticles;
		const float k = m_surfaceTension;

		unsigned int NrOfSamples;

		const unsigned int fluidModelIndex = sim->m_pointSetIndex;

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
				int numberOfNeighbours = sim->NumberOfNeighbors(fluidModelIndex, fluidModelIndex, i);

				if (numberOfNeighbours == 0)
				{
					m_mc_curv[i] = static_cast<float>(1.0) / supportRadius;
					continue;
				}

				const glm::vec3& xi = sim->m_x[i];

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
					const glm::vec3& xi = sim->m_x[i];
					glm::vec3 normalCorrection = glm::vec3(0,  0, 0);
					glm::vec3& ai = sim->m_a[i];


					float correctionForCurvature = 0;
					float correctionFactor = 0.0;

					glm::vec3 centroid = xi;
					glm::vec3 surfCentDir = glm::vec3(0,  0, 0);

					// collect neighbors
					std::multimap<float, size_t> neighs;

					glm::mat3x3 t = glm::mat3x3();
					int t_count = 0;
					glm::vec3 neighCent = glm::vec3(0,  0, 0);

					int nrNeighhbors = sim->NumberOfNeighbors(fluidModelIndex, fluidModelIndex, i);

					forall_fluid_neighbors_in_same_phase(
						if (m_mc_normals[neighborIndex] != glm::vec3(0,  0, 0))
						{
							glm::vec3& xj = sim->m_x[neighborIndex];
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
					if (m_mc_normals[i] != glm::vec3(0,  0, 0))
					{
						int count = 0;
						float CsCorr = 0.0;

						const glm::vec3& xi = sim->m_x[i];

						forall_fluid_neighbors_in_same_phase(
							if (m_mc_normals[neighborIndex] != glm::vec3(0,  0, 0))
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


							glm::vec3 final_normal = glm::vec3(0,  0, 0);
							float     final_curvature = m_mc_curv_smooth[i];

							final_normal = m_mc_normals_smooth[i];
							final_curvature = m_mc_curv_smooth[i];
							
							if (m_temporal_smoothing)
								m_final_curvatures[i] = static_cast<float>(0.05) * final_curvature + static_cast<float>(0.95) * m_final_curvatures_old[i];
							else
								m_final_curvatures[i] = final_curvature;

							glm::vec3 force = final_normal * k * m_final_curvatures[i];

							glm::vec3& ai = sim->m_a[i];
							ai -= (1 / sim->m_masses[i]) * force;

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
