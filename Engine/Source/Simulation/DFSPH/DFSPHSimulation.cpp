#include "pch.h"
#include "DFSPHSimulation.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Utility/SDF/MeshDistance.h"
#include "Core/Math/GaussQuadrature.h"

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
			// addDragMethod
			// addElasticityMethod
			m_surfaceTension = new SurfaceTensionZorillaRitter2020(this);
			m_viscosity = new ViscosityWeiler2018(this);
			// addVorticityMethod

			SetParticleRadius(particleRadius);

			m_neighborhoodSearch = new NeighborhoodSearch(m_supportRadius, false);
			m_neighborhoodSearch->SetRadius(m_supportRadius);
		}

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
	}

	DFSPHSimulation::~DFSPHSimulation()
	{
	}

	void DFSPHSimulation::OnUpdate()
	{
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
				auto dist_x = volumeMap->Interpolate(0u, x);

				if (dist_x > (1.0 + 1.0 /*/ factor*/) * supportRadius)
				{
					return 0.0;
				}

				auto integrand = [&](glm::vec3 const& xi) -> double
				{
					if (glm::dot(xi, xi) > supportRadius * supportRadius)
						return 0.0;

					auto dist = volumeMap->Interpolate(0u, x + xi);

					if (dist <= 0.0)
						return 1.0;// -0.001 * dist / supportRadius;
					if (dist < 1.0 / factor * supportRadius)
						return static_cast<double>(CubicKernel::W(factor * static_cast<float>(dist)) / CubicKernel::WZero());
					return 0.0;
				};

				double res = 0.0;
				res = 0.8 * GaussQuadrature::Integrate(integrand, int_domain, 30);

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
				if (dist == std::numeric_limits<double>::max())
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
		m_base = base;
	}

	SurfaceTensionZorillaRitter2020::SurfaceTensionZorillaRitter2020(DFSPHSimulation* base)
	{
		m_base = base;
	}
}
