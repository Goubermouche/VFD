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

#define forall_fluid_neighbors_in_same_phase(code) \
	for (unsigned int j = 0; j < m_base->NumberOfNeighbors(fluidModelIndex, fluidModelIndex, i); j++) \
	{ \
		const unsigned int neighborIndex = m_base->GetNeighbor(fluidModelIndex, fluidModelIndex, i, j); \
		const glm::vec3 &xj = m_base->m_x[neighborIndex]; \
		code \
	} 


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

		// Init the scene
		timeStepSize = 0.001;
		particleRadius = 0.025; 
		// Rigid bodies
		{
			BoundaryData* data = new BoundaryData();
			data->meshFile = "Resources/Models/Cube.obj";
			data->translation = { 0, -0.25, 0 };
			data->rotation = glm::quat();
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
		// Fluid models 
		{
			FluidData* data = new FluidData();
			data->samplesFile = "Resources/Models/Cube.obj";
			data->visMeshFile = "";
			data->id = "Fluid";
			data->translation = { 0.0, 1.8, -0.2 };
			data->rotation = glm::mat4(1.0f);
			data->scale = { 1, 1, 1 };
			data->initialVelocity = { 0, 0, 0 };
			data->initialAngularVelocity = { 0, 0, 0 };
			data->invert = false;
			data->resolutionSDF = { 20, 20, 20 };
			data->mode = 0;
			fluidModels.push_back(data);
		}
		// Materials 
		{
			MaterialData* data = new MaterialData();
			data->id = "Fluid";
			data->minVal = 0;
			data->maxVal = 10;
			data->colorField = "velocity";
			data->colorMapType = 1;
			data->maxEmitterParticles = 10000;
			data->emitterReuseParticles = false;
			data->emitterBoxMin = { -1, -1, -1 };
			data->emitterBoxMax = { 1, 1, 1 };
			materials.push_back(data);
		}

		// Init sim 
		{
			m_enableZSort = false;
			m_gravitation = { 0, -9.81, 0 };
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

		m_neighborhoodSearch->FindNeighbors();

		ClearAccelerations();
			
		{
			const unsigned int numParticles = m_numActiveParticles;
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++) {
					if (m_particleState[i] == ParticleState::Active) {
						glm::vec3& vel = m_v[i];
						vel += h * m_a[i];
					}
				}
			}
		}
	
		{
			const unsigned int numParticles = m_numActiveParticles;
			#pragma omp parallel default(shared)
			{
				#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					if (m_particleState[i] == ParticleState::Active)
					{
						glm::vec3& xi = m_x[i];
						const glm::vec3& vi = m_v[i];
						xi += h * vi;
					}
				}
			}
		}


	}

	void DFSPHSimulation::OnRenderTemp()
	{
		for (size_t i = 0; i < m_numParticles; i++)
		{
			Renderer::DrawPoint(m_x[i], { 0.7, 0.7 , 0.7, 1}, particleRadius * 35);
		}
	}

	void DFSPHSimulation::InitVolumeMap(std::vector<glm::vec3>& x, std::vector<glm::ivec3>& faces, const BoundaryData* boundaryData, const bool md5, const bool isDynamic, BoundaryModelBender2019* boundaryModel)
	{
		SDF* volumeMap;
		glm::ivec3 resolutionSDF = boundaryData->mapResolution;
		const float supportRadius = m_supportRadius;

		{
			//////////////////////////////////////////////////////////////////////////
			// Generate distance field of object using Discregrid
			//////////////////////////////////////////////////////////////////////////

			std::vector<glm::vec3> doubleVec;
			doubleVec.resize(x.size());
			for (unsigned int i = 0; i < x.size(); i++) {
				doubleVec[i] = { x[i].x, x[i].y , x[i].z };
			}
			EdgeMesh sdfMesh(doubleVec, faces);

			MeshDistance md(sdfMesh);
			BoundingBox domain;
			for (auto const& x_ : x)
			{
				domain.Extend(x_);
			}

			const float tolerance = boundaryData->mapThickness;
			domain.max += (8.0f * supportRadius + tolerance) * domain.Diagonal();
			domain.min -= (8.0f * supportRadius + tolerance) * domain.Diagonal();

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
				auto x = glm::max(x_, glm::min(volumeMap->GetDomain().min, volumeMap->GetDomain().max));
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

		//Poly6Kernel::setRadius(m_supportRadius);
		//SpikyKernel::setRadius(m_supportRadius);
		//CubicKernel::setRadius(m_supportRadius);
		//WendlandQuinticC2Kernel::setRadius(m_supportRadius);
		//PrecomputedCubicKernel::setRadius(m_supportRadius);
		//CohesionKernel::setRadius(m_supportRadius);
		//AdhesionKernel::setRadius(m_supportRadius);
		//CubicKernel2D::setRadius(m_supportRadius);
		//WendlandQuinticC2Kernel2D::setRadius(m_supportRadius);
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
			for (size_t i = 0; i < m_numParticles; i++)
			{
				const glm::vec3& xi = m_x[i];
				ComputeVolumeAndBoundaryX(i, xi);
			}
		}
	}

	void DFSPHSimulation::ComputeVolumeAndBoundaryX(const unsigned int i, const glm::vec3& xi)
	{
		const float supportRadius =  m_supportRadius;
		const float particleRadius = particleRadius;
		const float dt = timeStepSize;

		BoundaryModelBender2019* bm = m_boundaryModels;
		glm::vec3& boundaryXj = bm->GetBoundaryXj(i);
		boundaryXj = { 0, 0, 0 };
		float& boundaryVolume = bm->GetBoundaryVolume(i);
		boundaryVolume = 0.0;

		const glm::vec3 t = bm->GetRigidBody()->m_x;
		const glm::mat3 R = glm::toMat3(bm->GetRigidBody()->m_q);

		glm::vec3 normal;
		const glm::vec3 localXi = (glm::transpose(R) * (xi - t));

		std::array<unsigned int, 32> cell;
		glm::vec3 c0;
		std::array<float, 32> N;
		std::array<std::array<float, 3>, 32> dN;
		bool chk = bm->m_map->DetermineShapeFunctions(0, localXi, cell, c0, N, &dN);

		float dist = std::numeric_limits<float>::max();
		if (chk) {
			dist = bm->m_map->Interpolate(0, localXi, cell, c0, N, &normal, &dN);
		}

		bool animateParticle = false;
		if (m_particleState[i] == ParticleState::Active) {
			if ((dist > 0.0) && (static_cast<float>(dist) < supportRadius)) {
				const float volume = bm->m_map->Interpolate(1, localXi, cell, c0, N);
				if ((volume > 0.0) && (volume != std::numeric_limits<float>::max())) {
					boundaryVolume = static_cast<float>(volume);

					normal = R * normal;
					const double nl = std::sqrt(glm::dot(normal, normal));
					if (nl > 1.0e-9)
					{
						normal /= nl;
						const float d = glm::max((static_cast<float>(dist) + static_cast<float>(0.5) * particleRadius), static_cast<float>(2.0) * particleRadius);
						boundaryXj = (xi - d * normal);
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
				ERR("Particle in boundary.");
				// if a particle is in the boundary, animate the particle back
				animateParticle = true;
				boundaryVolume = 0.0;
			}
			else
			{
				boundaryVolume = 0.0;
			}
		}

		if (animateParticle)
		{
			if (dist != std::numeric_limits<float>::max())				// if dist is numeric_limits<double>::max(), then the particle is not close to the current boundary
			{
				normal = R * normal;
				const double nl = std::sqrt(glm::dot(normal, normal));

				if (nl > 1.0e-5)
				{
					normal /= nl;
					// project to surface
					float delta = static_cast<float>(2.0) * particleRadius - static_cast<float>(dist);
					delta = std::min(delta, static_cast<float>(0.1) * particleRadius);		// get up in small steps
					m_x[i] = (xi + delta * normal);
					// adapt velocity in normal direction
					//model->getVelocity(i) = 1.0/dt * delta * normal.cast<Real>();
					m_v[i] = { 0, 0, 0 };
				}
			}
			boundaryVolume = 0.0;
		}
	}

	void DFSPHSimulation::ComputeDensities()
	{
		const float density0 = m_density0;
		const unsigned int numParticles = m_numActiveParticles;

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				float& density = m_density[i];
				density = m_V * m_W_zero;
				const glm::vec3& xi = m_x[i];

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors(
					density += m_V * PrecomputedCubicKernel::W(xi - xj);
				);

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				//forall_volume_maps(
				//	density += Vj * PrecomputedCubicKernel::W(xi - xj);
				//);

				density *= density0;
			}
		}
	}

	void DFSPHSimulation::ComputeDFSPHFactor()
	{
		const int numParticles = m_numActiveParticles;
		#pragma omp parallel default(shared)
		{
			//////////////////////////////////////////////////////////////////////////
			// Compute pressure stiffness denominator
			//////////////////////////////////////////////////////////////////////////
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++)
			{
				//////////////////////////////////////////////////////////////////////////
				// Compute gradient dp_i/dx_j * (1/k)  and dp_j/dx_j * (1/k)
				//////////////////////////////////////////////////////////////////////////

				const glm::vec3& xi = m_x[i];
				float sum_grad_p_k = 0.0;
				glm::vec3 grad_p_i = { 0, 0, 0 };

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors(
					const glm::vec3 grad_p_j = -m_V * PrecomputedCubicKernel::GradientW(xi - xj);
					sum_grad_p_k += glm::dot(grad_p_j, grad_p_j);
					grad_p_i -= grad_p_j;
				);

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				//forall_volume_maps(
				//	const glm::vec3 grad_p_j = -Vj * PrecomputedCubicKernel::GradientW(xi - xj);
				//	grad_p_i -= grad_p_j;
				//);

				sum_grad_p_k += glm::dot(grad_p_i, grad_p_i);

				//////////////////////////////////////////////////////////////////////////
				// Compute pressure stiffness denominator
				//////////////////////////////////////////////////////////////////////////
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

		//////////////////////////////////////////////////////////////////////////
		// Compute velocity of density change
		//////////////////////////////////////////////////////////////////////////

		const int numParticles = (int)m_numActiveParticles;
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++) {
				ComputeDensityChange(i, h);
				m_simulationData.GetFactor(0, i) *= invH;
			}
		}

		m_iterationsV = 0;
		//////////////////////////////////////////////////////////////////////////
		// Start solver
		//////////////////////////////////////////////////////////////////////////

		float avg_density_err = 0.0;
		bool chk = false;
		while ((!chk || (m_iterationsV < 1)) && (m_iterationsV < maxIter)) {
			chk = true;

			const float density0 = m_density0;
			avg_density_err = 0.0;
			DivergenceSolveIteration(0, avg_density_err);

			const float eta = (static_cast<float>(1.0) / h) * maxError * static_cast<float>(0.01) * density0;  // maxError is given in percent
			chk = chk && (avg_density_err <= eta);
			m_iterationsV++;
		}

		//////////////////////////////////////////////////////////////////////////
		// Multiply by h, the time step size has to be removed 
		// to make the stiffness value independent 
		// of the time step size
		//////////////////////////////////////////////////////////////////////////

		for (int i = 0; i < numParticles; i++) {
			m_simulationData.GetKappaV(0, i) *= h;
		}

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
		const int numParticles = m_numActiveParticles;
		if (numParticles == 0) {
			return;
		}

		#pragma omp parallel default(shared)
		{
			//////////////////////////////////////////////////////////////////////////
			// Divide by h^2, the time step size has been removed in 
			// the last step to make the stiffness value independent 
			// of the time step size
			//////////////////////////////////////////////////////////////////////////
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++) {
				ComputeDensityChange(i, h);
				if (m_simulationData.GetDensityAdv(0, i) > 0.0) {
					m_simulationData.GetKappaV(0, i) = static_cast<float>(0.5) * std::max(m_simulationData.GetKappaV(0, i), static_cast<float>(-0.5)) * invH;
				}
				else {
					m_simulationData.GetKappaV(0, i) = 0.0;
				}
			}

			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++) {
				if (m_particleState[i] != ParticleState::Active)
				{
					m_simulationData.GetKappaV(0, i) = 0.0;
					continue;
				}

				//if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 0.0)
				{
					glm::vec3& vel = m_v[i];
					const float ki = m_simulationData.GetKappaV(0, i);
					const glm::vec3& xi = m_x[i];

					//////////////////////////////////////////////////////////////////////////
					// Fluid
					//////////////////////////////////////////////////////////////////////////
					forall_fluid_neighbors(
						const float kj = m_simulationData.GetKappaV(0, neighborIndex);

						const float kSum = (ki + m_density0 / density0 * kj);
						if (fabs(kSum) > m_eps) {
							const glm::vec3 grad_p_j = -m_V * PrecomputedCubicKernel::GradientW(xi - xj);
						}
					);

					//////////////////////////////////////////////////////////////////////////
					// Boundary
					//////////////////////////////////////////////////////////////////////////
					if (fabs(ki) > m_eps) {
						//forall_volume_maps(
						//	const glm::vec3 grad_p_j = -Vj * PrecomputedCubicKernel::GradientW(xi - xj);
						//	const glm::vec3 velChange = -h * (float)1.0 * ki * grad_p_j; // kj already contains inverse density
						//	vel += velChange;
						//)
					}
				}
			}
		}
	}

	void DFSPHSimulation::ComputeDensityChange(const unsigned int i, const float h)
	{
		float& densityAdv = m_simulationData.GetDensityAdv(0, i);
		const glm::vec3& xi = m_x[i];
		const glm::vec3& vi = m_v[i];
		densityAdv = 0.0;
		unsigned int numNeighbors = 0;

		//////////////////////////////////////////////////////////////////////////
		// Fluid
		//////////////////////////////////////////////////////////////////////////
		//forall_fluid_neighbors(
		//	const glm::vec3 & vj = fm_neighbor->getVelocity(neighborIndex);
		//	// densityAdv += fm_neighbor->getVolume(neighborIndex) * (vi - vj).dot(sim->gradW(xi - xj));
		//);

		//////////////////////////////////////////////////////////////////////////
		// Boundary
		//////////////////////////////////////////////////////////////////////////
		//forall_volume_maps(
		//	glm::vec3 vj;
		//	bm_neighbor->getPointVelocity(xj, vj);
		//	// densityAdv += Vj * (vi - vj).dot(sim->gradW(xi - xj));
		//);

		densityAdv = std::max(densityAdv, static_cast<float>(0.0));
		for (unsigned int pid = 0; pid < m_neighborhoodSearch->GetPointSetCount(); pid++) {
			numNeighbors +NumberOfNeighbors(0, pid, i);
		}

		if (numNeighbors < 20) {
			densityAdv = 0.0;
		}
	}

	void DFSPHSimulation::DivergenceSolveIteration(const unsigned int fluidModelIndex, float& avg_density_err)
	{
		const float density0 = m_density0;
		const int numParticles = (int)m_numActiveParticles;
		if (numParticles == 0)
			return;

		const float h = timeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		float density_error = 0.0;
		//////////////////////////////////////////////////////////////////////////
		// Perform Jacobi iteration over all blocks
		//////////////////////////////////////////////////////////////////////////	
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++) {
				if (m_particleState[i] != ParticleState::Active)
					continue;

				//////////////////////////////////////////////////////////////////////////
				// Evaluate rhs
				//////////////////////////////////////////////////////////////////////////
				const float b_i = m_simulationData.GetDensityAdv(fluidModelIndex, i);
				const float ki = b_i * m_simulationData.GetFactor(fluidModelIndex, i);
				m_simulationData.GetKappaV(fluidModelIndex, i) += ki;

				glm::vec3& v_i = m_v[i];
				const glm::vec3& xi = m_x[i];

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors(
					const float b_j = m_simulationData.GetDensityAdv(0, neighborIndex);
					const float kj = b_j * m_simulationData.GetFactor(0, neighborIndex);

					const float kSum = ki + m_density0 / density0 * kj;
					if (fabs(kSum) > m_eps) {
						const glm::vec3 grad_p_j = -m_V * PrecomputedCubicKernel::GradientW(xi - xj);
						v_i -= h * kSum * grad_p_j;			// ki, kj already contain inverse density
					}
				);

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				if (fabs(ki) > m_eps) {
					//forall_volume_maps(
					//	const glm::vec3 grad_p_j = -Vj * PrecomputedCubicKernel::GradientW(xi - xj);
					//	const glm::vec3 velChange = -h * (float)1.0 * ki * grad_p_j;				// kj already contains inverse density
					//	v_i += velChange;

					//	m_boundaryModels->AddForce(xj, -m_masses[i] * velChange * invH);
					//	// bm_neighbor->addForce(xj, -model->getMass(i) * velChange * invH);
					//);
				}
			}

			//////////////////////////////////////////////////////////////////////////
			// Update rho_adv and density error
			//////////////////////////////////////////////////////////////////////////
			#pragma omp for reduction(+:density_error) schedule(static)
			for (int i = 0; i < (int)numParticles; i++) {
				ComputeDensityChange(i, h);
				density_error += density0 * m_simulationData.GetDensityAdv(fluidModelIndex, i);
			}
		}

		avg_density_err = density_error / numParticles;
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
		// 1 - updateTimeStepSizeCFL();
		const float radius = particleRadius;
		float h = timeStepSize;
		float maxVel = 0.1;
		const float diameter = static_cast<float>(2.0) * radius;

		// fluid particles
		const unsigned int numParticles = m_numActiveParticles;
		for (unsigned int i = 0; i < numParticles; i++)
		{
			const glm::vec3& vel = m_v[i];
			const glm::vec3& accel = m_a[i];
			const float velMag = glm::dot((vel + accel * h), (vel + accel * h));
			if (velMag > maxVel)
				maxVel = velMag;
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

		//////////////////////////////////////////////////////////////////////////
		// Compute rho_adv
		//////////////////////////////////////////////////////////////////////////
		const float density0 = m_density0;
		const int numParticles = (int)m_numActiveParticles;
		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++) {
				ComputeDensityAdv( i, numParticles, h, density0);
				m_simulationData.GetFactor(0, i) *= invH2;
			}
		}

		m_iterations = 0;

		//////////////////////////////////////////////////////////////////////////
		// Start solver
		//////////////////////////////////////////////////////////////////////////

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
		const int numParticles = m_numParticles;

		if (numParticles == 0)
			return;

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				//m_simulationData.getKappa(fluidModelIndex, i) = max(m_simulationData.getKappa(fluidModelIndex, i)*invH2, -static_cast<Real>(0.5) * density0*density0);
				ComputeDensityAdv(i, numParticles, h, density0);
				if (m_simulationData.GetDensityAdv(0, i) > 1.0)
					m_simulationData.GetKappa(0, i) = static_cast<float>(0.5) * std::max(m_simulationData.GetKappa(0, i), static_cast<float>(-0.00025)) * invH2;
				else
					m_simulationData.GetKappa(0, i) = 0.0;
			}

			#pragma omp for schedule(static)  
			for (int i = 0; i < numParticles; i++) {
				if (m_particleState[i] != ParticleState::Active)
				{
					m_simulationData.GetKappa(0, i) = 0.0;
					continue;
				}

				//if (m_simulationData.getDensityAdv(fluidModelIndex, i) > 1.0)
				{
					glm::vec3& vel = m_v[i];
					const float ki = m_simulationData.GetKappa(0, i);
					const glm::vec3& xi = m_x[i];

					forall_fluid_neighbors(
						const float kj = m_simulationData.GetKappa(0, neighborIndex);
						const float kSum = (ki + m_density0 / density0 * kj);
						if (fabs(kSum) > m_eps) {
							const  glm::vec3 grad_p_j = -m_V * PrecomputedCubicKernel::GradientW(xi - xj);
						}
					);

					//forall_volume_maps(
					//	const glm::vec3 grad_p_j = -Vj * PrecomputedCubicKernel::GradientW(xi - xj);
					//	const glm::vec3 velChange = -h * (float)1.0 * ki * grad_p_j;				// kj already contains inverse density
					//	vel += velChange;

					//	// m_boundaryHandlingMethod->Add(xj, -model->getMass(i) * velChange * invH);
					//);
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

		forall_fluid_neighbors(
			const glm::vec3 & vj = m_v[neighborIndex];
			delta += m_V * glm::dot(vi - vj, PrecomputedCubicKernel::GradientW(xi - xj));
		);

		//forall_volume_maps(
		//	glm::vec3 vj;
		//	m_boundaryModels->GetPointVelocity(xj, vj);
		//	delta += Vj * glm::dot((vi - vj), PrecomputedCubicKernel::GradientW(xi - xj));
		//)

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

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static) 
			for (int i = 0; i < numParticles; i++) {
				if (m_particleState[i] != ParticleState::Active)
					continue;

				const float b_i = m_simulationData.GetDensityAdv(0, i) - static_cast<float>(1.0);
				const float ki = b_i * m_simulationData.GetFactor(0, i);

				m_simulationData.GetKappa(0, i) += ki;

				glm::vec3& v_i = m_v[i];
				const glm::vec3& xi = m_x[i];

				forall_fluid_neighbors(
					const float b_j = m_simulationData.GetDensityAdv(0, neighborIndex) - static_cast<float>(1.0);
					const float kj = b_j * m_simulationData.GetFactor(0, neighborIndex);
					const float kSum = ki + m_density0 / density0 * kj;

					if (fabs(kSum) > m_eps) {
						const glm::vec3 grad_p_j = -m_V * PrecomputedCubicKernel::GradientW(xi - xj);
						v_i -= h * kSum * grad_p_j;
					}
				);

				if (fabs(ki) > m_eps) {
					//forall_volume_maps(
					//	const glm::vec3 grad_p_j = -Vj * PrecomputedCubicKernel::GradientW(xi - xj);
					//	const glm::vec3 velChange = -h * (float)1.0 * ki * grad_p_j;
					//	v_i += velChange;

					//	// bm_neighbor->addForce(xj, -model->getMass(i) * velChange * invH);
					//);
				}
			}

			//////////////////////////////////////////////////////////////////////////
			// Update rho_adv and density error
			//////////////////////////////////////////////////////////////////////////

			#pragma omp for reduction(+:density_error) schedule(static) 
			for (int i = 0; i < numParticles; i++)
			{
				ComputeDensityAdv(i, numParticles, h, density0);

				density_error += density0 * m_simulationData.GetDensityAdv(0, i) - density0;
			}
		}
	}

	void DFSPHSimulation::InitFluidData()
	{
		m_density0 = 1000;
		float diam = static_cast<float>(2.0) * particleRadius;
		m_V = static_cast<float>(0.8) * diam * diam * diam;

		EdgeMesh mesh("Resources/Models/Cube.obj", { .3,  .3, .3 });
		for (const glm::vec3& sample : ParticleSampler::SampleMeshVolume(mesh, particleRadius, {20, 20, 20}, false, SampleMode::MaxDensity))
		{
			m_x.push_back({sample + glm::vec3{0, 3, 0}});
			m_v.push_back({ 0, 0, 0 });

			m_x0.push_back(m_x.back());
			m_v0.push_back(m_v.back());
			m_a.push_back({ 0, 0, 0 });
			m_density.push_back(0);
			m_particleState.push_back(ParticleState::Active);
			m_masses.push_back(m_V * m_density0);
		}

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
		// m_viscosity->deferredInit();
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

	ViscosityWeiler2018::ViscosityWeiler2018(DFSPHSimulation* base)
	{
		m_maxIter = 100;
		m_maxError = static_cast<float>(0.01);
		m_iterations = 0;
		m_boundaryViscosity = 0.0;
		m_tangentialDistanceFactor = static_cast<float>(0.5);

		m_vDiff.resize(base->m_numParticles, glm::vec3(0, 0, 0));
		m_base = base;
	}

	void ViscosityWeiler2018::OnUpdate() {
		const int numParticles = m_base->m_numActiveParticles;
		// prevent solver from running with a zero-length vector
		if (numParticles == 0)
			return;
		const float density0 = m_base->m_density0;
		const float h = m_base->timeStepSize;
		//////////////////////////////////////////////////////////////////////////
		// Init linear system solver and preconditioner
		//////////////////////////////////////////////////////////////////////////

		MatrixReplacement A(3 * m_base->m_numActiveParticles, MatrixVecProd, (void*)this, m_base);
	}

	void ViscosityWeiler2018::MatrixVecProd(const float* vec, float* result, void* userData, DFSPHSimulation* m_base)
	{
		ViscosityWeiler2018* visco = (ViscosityWeiler2018*)userData;
		const unsigned int numParticles = m_base->m_numActiveParticles;
		const unsigned int fluidModelIndex = m_base->m_pointSetIndex;

		const float h = m_base->m_supportRadius;
		const float h2 = h * h;
		const float dt = m_base->timeStepSize;
		const float density0 = m_base->m_density0;
		const float mu = visco->m_viscosity * density0;
		const float mub = visco->m_boundaryViscosity * density0;
		const float sphereVolume = static_cast<float>(4.0 / 3.0 * PI) * h2 * h;

		float d = 10.0;

		BoundaryModelBender2019* m_boundaryModels = m_base->m_boundaryModels;


#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static) 
			for (int i = 0; i < (int)numParticles; i++)
			{
				const glm::vec3& xi = m_base->m_x[i];
				glm::vec3 ai;
				ai = { 0, 0, 0 };
				const float density_i = m_base->m_density[i];
				const glm::vec3& vi = glm::vec3(vec[i * 3 + 0], vec[i * 3 + 1], vec[i * 3 + 2]);

				//////////////////////////////////////////////////////////////////////////
				// Fluid
				//////////////////////////////////////////////////////////////////////////
				forall_fluid_neighbors_in_same_phase(
					const float density_j = m_base->m_density[neighborIndex];
					const glm::vec3 gradW = PrecomputedCubicKernel::GradientW(xi - xj);

					const glm::vec3& vj = glm::vec3(vec[neighborIndex * 3 + 0], vec[neighborIndex * 3 + 1], vec[neighborIndex * 3 + 2]);
					const glm::vec3 xixj = xi - xj;
					ai += d * mu * (m_base->m_masses[neighborIndex] / density_j) * glm::dot(vi - vj, xixj) / (glm::dot(xixj, xixj) + +0.01f * h2) * gradW;
				);

				//////////////////////////////////////////////////////////////////////////
				// Boundary
				//////////////////////////////////////////////////////////////////////////
				if (mub != 0.0)
				{
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

							// each sample point represents the quarter of the volume inside of the boundary
							const float vol = static_cast<float>(0.25) * Vj;

							glm::vec3 v1(0, 0, 0);
							glm::vec3 v2(0, 0, 0);
							glm::vec3 v3(0, 0, 0);
							glm::vec3 v4(0, 0, 0);

							const glm::vec3 a1 = d * mub * vol * glm::dot(vi, xix1) / (glm::dot(xix1, xix1) + 0.01f * h2) * gradW1;
							const glm::vec3 a2 = d * mub * vol * glm::dot(vi, xix2) / (glm::dot(xix2, xix2) + 0.01f * h2) * gradW2;
							const glm::vec3 a3 = d * mub * vol * glm::dot(vi, xix3) / (glm::dot(xix3, xix3) + 0.01f * h2) * gradW3;
							const glm::vec3 a4 = d * mub * vol * glm::dot(vi, xix4) / (glm::dot(xix4, xix4) + 0.01f * h2) * gradW4;
							ai += a1 + a2 + a3 + a4;
						}
					);
				}

				result[3 * i] = vec[3 * i] - dt / density_i * ai[0];
				result[3 * i + 1] = vec[3 * i + 1] - dt / density_i * ai[1];
				result[3 * i + 2] = vec[3 * i + 2] - dt / density_i * ai[2];
			}
		}
	}

	SurfaceTensionZorillaRitter2020::SurfaceTensionZorillaRitter2020(DFSPHSimulation* base)
		: 
		m_surfaceTension(0.05), 
		m_surfaceTensionBoundary(0.01)
		, m_Csd(10000) // 10000 // 36000 // 48000 // 60000
		, m_tau(0.5)
		, m_r2mult(0.8)
		, m_r1(base->m_supportRadius)
		, m_r2(m_r2mult* m_r1)
		, m_class_k(74.688796680497925)
		, m_class_d(12)
		, m_temporal_smoothing(false)
		, m_CsdFix(-1)
		, m_class_d_off(2)
		, m_pca_N_mix(0.75)
		, m_pca_C_mix(0.5)
		, m_neighs_limit(16)
		, m_CS_smooth_passes(1)
		, m_halton_sampling(RandomMethod::HALTON)
		, m_normal_mode(NormalMethod::MC)

	{
		m_base = base;
		m_mc_normals.resize(base->m_numParticles, { 0, 0, 0 });
		m_final_curvatures.resize(base->m_numParticles, 0.0);

		m_pca_curv.resize(base->m_numParticles, 0.0);
		m_pca_curv_smooth.resize(base->m_numParticles, 0.0);
		m_mc_curv.resize(base->m_numParticles, 0.0);
		m_mc_curv_smooth.resize(base->m_numParticles, 0.0);

		m_mc_normals_smooth.resize(base->m_numParticles, { 0, 0, 0 });
		m_pca_normals.resize(base->m_numParticles, { 0, 0, 0 });

		m_final_curvatures_old.resize(base->m_numParticles, 0.0);

		m_classifier_input.resize(base->m_numParticles, 0.0);

		m_classifier_output.resize(base->m_numParticles, 0.0);
	}

	void SurfaceTensionZorillaRitter2020::OnUpdate()
	{
		// Step ritter
		float timeStep = m_base->timeStepSize;
		m_r2 = m_r1 * m_r2mult;
		const float supportRadius = m_base->m_supportRadius;
		const unsigned int numParticles = m_base->m_numActiveParticles;
		const float k = m_surfaceTension;

		unsigned int NrOfSamples;
		const unsigned int fluidModelIndex = m_base->m_pointSetIndex;

		if (m_CsdFix > 0)
			NrOfSamples = m_CsdFix;
		else
			NrOfSamples = int(m_Csd * timeStep);

		// ################################################################################################
		// ## first pass, compute classification and first estimation for normal and curvature (Montecarlo)
		// ################################################################################################

#pragma omp parallel default(shared)
		{
#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				// init or reset arrays
				m_mc_normals[i] = { 0, 0, 0 };

				m_pca_normals[i] = { 0, 0, 0 };
				m_mc_normals_smooth[i] = { 0, 0, 0 };

				m_mc_curv[i] = 0.0;
				m_mc_curv_smooth[i] = 0.0;
				m_pca_curv[i] = 0.0;
				m_pca_curv_smooth[i] = 0.0;
				m_final_curvatures[i] = 0.0;

				// -- compute center of mass of current particle
				glm::vec3 centerofMasses = { 0, 0, 0 };
				int numberOfNeighbours = m_base->NumberOfNeighbors(fluidModelIndex, fluidModelIndex, i);

				if (numberOfNeighbours == 0)
				{
					m_mc_curv[i] = static_cast<float>(1.0) / supportRadius;
					continue;
				}

				const glm::vec3& xi = m_base->m_x[i];

				forall_fluid_neighbors_in_same_phase(
					glm::vec3 xjxi = (xj - xi);
					centerofMasses += xjxi;
				);

				centerofMasses /= supportRadius;

				// cache classifier input, could also be recomputed later to avoid caching
				m_classifier_input[i] = std::sqrt(glm::dot(centerofMasses, centerofMasses)) / static_cast<float>(numberOfNeighbours);

				// -- if it is a surface classified particle
				if (ClassifyParticleConfigurable(m_classifier_input[i], numberOfNeighbours)) { //EvaluateNetwork also possible
					// -- create monte carlo samples on particle
					std::vector<glm::vec3> points;

					// HALTON
					points = GetSphereSamplesLookUp(NrOfSamples, supportRadius, i * NrOfSamples, haltonVec323, static_cast<int>(haltonVec323.size())); // 8.5 // 15.0(double) // 9.0(float)

					forall_fluid_neighbors_in_same_phase(
						glm::vec3 xjxi = (xj - xi);
						for (int p = static_cast<int>(points.size()) - 1; p >= 0; --p)
						{
							glm::vec3 vec = (points[p] - xjxi);
							float dist = glm::dot(vec, vec);

							if (dist <= pow((m_r2 / m_r1), 2) * supportRadius * supportRadius)
								points.erase(points.begin() + p);
						}
					);

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
						m_mc_normals[i] = { 0, 0, 0 };
						m_mc_curv[i] = 0.0;
						m_classifier_output[i] = 0.5; // -- used for visualize post-correction points (white in the paper)
					}
				}
				else {
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
			for (int i = 0; i < (int)numParticles; i++) {
				if (m_mc_normals[i] != glm::vec3(0, 0, 0)) {
					const glm::vec3& xi = m_base->m_x[i];
					glm::vec3 normalCorrection = { 0, 0, 0 };
					glm::vec3& ai = m_base->m_a[i];

					float correctionForCurvature = 0;
					float correctionFactor = 0.0;

					glm::vec3 centroid = xi;
					glm::vec3 surfCentDir = { 0, 0, 0 };

					// collect neighbors
					std::multimap<float, size_t> neighs;

					glm::mat3x3 t = glm::mat3x3(0.0);
					int t_count = 0;
					glm::vec3 neighCent = { 0, 0, 0 };

					int nrNeighhbors = m_base->NumberOfNeighbors(fluidModelIndex, fluidModelIndex, i);

					forall_fluid_neighbors_in_same_phase(
						if (m_mc_normals[neighborIndex] != glm::vec3(0, 0, 0)) {
							glm::vec3& xj = m_base->m_x[neighborIndex];
							glm::vec3 xjxi = (xj - xi);

							surfCentDir += xjxi;
							centroid += xj;
							t_count++;

							float sum = 0;
							float distanceji = std::sqrt(glm::dot(xjxi, xjxi));

							normalCorrection += m_mc_normals[neighborIndex] * (1 - distanceji / supportRadius);
							correctionForCurvature += m_mc_curv[neighborIndex] * (1 - distanceji / supportRadius);
							correctionFactor += (1 - distanceji / supportRadius);
						}
						else if (m_normal_mode != NormalMethod::MC
							&& ClassifyParticleConfigurable(m_classifier_input[neighborIndex], nrNeighhbors, m_class_d_off))
						{
							glm::vec3& xj = m_base->m_x[neighborIndex];
							glm::vec3 xjxi = (xj - xi);
							surfCentDir += xjxi;
							centroid += xj;
							t_count++;
							float distanceji = std::sqrt(glm::dot(xjxi, xjxi));
							neighs.insert({ distanceji, neighborIndex });
						}
					);


					normalCorrection = glm::normalize(normalCorrection);
					m_mc_normals_smooth[i] = (1 - m_tau) * m_mc_normals[i] + m_tau * normalCorrection;
					m_mc_normals_smooth[i] = glm::normalize(m_mc_normals_smooth[i]);
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
					if (m_mc_normals[i] != glm::vec3(0, 0, 0)) {
						int count = 0;
						float CsCorr = 0.0;

						const glm::vec3& xi = m_base->m_x[i];

						forall_fluid_neighbors_in_same_phase(
							if (m_mc_normals[neighborIndex] != glm::vec3(0, 0, 0)) {
								CsCorr += m_pca_curv[neighborIndex];
								count++;
							}
						);

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

						glm::vec3 final_normal = { 0, 0, 0 };
						float     final_curvature = m_mc_curv_smooth[i];

						final_normal = m_mc_normals_smooth[i];
						final_curvature = m_mc_curv_smooth[i];

						if (m_temporal_smoothing)
							m_final_curvatures[i] = static_cast<float>(0.05) * final_curvature + static_cast<float>(0.95) * m_final_curvatures_old[i];
						else
							m_final_curvatures[i] = final_curvature;

						glm::vec3 force = final_normal * k * m_final_curvatures[i];

						glm::vec3& ai = m_base->m_a[i];
						ai -= (1 / m_base->m_masses[i]) * force;

						m_final_curvatures_old[i] = m_final_curvatures[i];
					}
					else {
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
		double neighborsOnTheLine = m_class_k * com + m_class_d + d_offset; // pre-multiplied

		if (non <= neighborsOnTheLine)
			return true;
		else
			return false;
	}

	std::vector<glm::vec3> SurfaceTensionZorillaRitter2020::GetSphereSamplesLookUp(int N, float supportRadius, int start, const std::vector<float>& vec3, int mod)
	{
		std::vector<glm::vec3> points(N);
		int s = (start / 3) * 3; // ensure to be dividable by 3
		for (int i = 0; i < N; i++)
		{
			int i3 = s + 3 * i;
			points[i] = supportRadius * glm::vec3(vec3[i3 % mod], vec3[(i3 + 1) % mod], vec3[(i3 + 2) % mod]);
		}
		return points;
	}
}
