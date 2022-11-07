#include "pch.h"
#include "DFSPHSimulation.h"
#include "Utility/Sampler/ParticleSampler.h"
#include "Utility/SDF/MeshDistance.h"
#include "Core/Math/GaussQuadrature.h"
#include "Core/Math/HaltonVec323.h"

#define ForAllFluidNeighbors(code) \
	for (unsigned int j = 0; j < NumberOfNeighbors(0, 0, i); j++) \
	{ \
		const unsigned int neighborIndex = GetNeighbor(0, 0, i, j); \
		const glm::vec3 &xj = m_ParticlePositions[neighborIndex]; \
		code \
	} \

#define ForAllFluidNeighborsAVX(code)\
	const unsigned int maxN = m_Base->NumberOfNeighbors(0, 0, i); \
	for (unsigned int j = 0; j < maxN; j += 8) \
	{ \
		const unsigned int count = std::min(maxN - j, 8u); \
		const Scalar3f8 xj_avx = ConvertScalarZero(&m_Base->GetNeighborList(0, 0, i)[j], &m_ParticlePositions[0], count); \
		code \
	} \

#define ForAllFluidNeighborsInSamePhase(code) \
	for (unsigned int j = 0; j < m_Base->NumberOfNeighbors(0, 0, i); j++) \
	{ \
		const unsigned int neighborIndex = m_Base->GetNeighbor(0, 0, i, j); \
		const glm::vec3 &xj = m_Base->GetParticlePosition(neighborIndex); \
		code \
	} 

#define ForAllFluidNeighborsAVXNOX(code) \
	unsigned int idx = 0; \
	const unsigned int maxN = m_Base->NumberOfNeighbors(0, 0, i); \
	for (unsigned int j = 0; j < maxN; j += 8) \
	{ \
		const unsigned int count = std::min(maxN - j, 8u); \
		code \
		idx++; \
	} \

#define ForAllFluidNeighborsInSamePhaseAVX(code) \
    const unsigned int maxN = sim->NumberOfNeighbors(0, 0, i);  \
    for (unsigned int j = 0; j < maxN; j += 8) \
    { \
		const unsigned int count = std::min(maxN - j, 8u); \
		const Scalar3f8 xj_avx = ConvertScalarZero(&sim->GetNeighborList(0, 0, i)[j], &sim->GetParticlePosition(0), count); \
		code \
	} \


#define ForAllVolumeMaps(code) \
	for(unsigned int pid = 0; pid < m_RigidBodies.size(); pid++) { \
		StaticRigidBody* bm_neighbor = m_RigidBodies[pid]; \
		const float Vj = bm_neighbor->GetBoundaryVolume(i);  \
		if (Vj > 0.0) \
		{ \
			const glm::vec3 &xj = bm_neighbor->GetBoundaryXJ(i); \
			code \
		} \
	} \
	
#define ComputeVJGradientW() const Scalar3f8& V_gradW = m_PrecalculatedVolumeGradientW[m_PrecalculatedIndices[i] + idx];
#define ComputeVJGradientSamePhase() const Scalar3f8& V_gradW = sim->GetPrecalculatedVolumeGradientW(sim->GetPrecalculatedIndicesSamePhase(i) + j / 8);

namespace fe {
	DFSPHSimulation::DFSPHSimulation(const DFSPHSimulationDescription& desc)
		: m_Description(desc)
	{
		m_Material = Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl"));
		m_Material->Set("color", { 0.4f, 0.4f, 0.4f });

		SetParticleRadius(m_Description.ParticleRadius);
		m_NeighborhoodSearch = new NeighborhoodSearch(m_SupportRadius, false);
		m_NeighborhoodSearch->SetRadius(m_SupportRadius);
		m_WZero = PrecomputedCubicKernel::WZero();
		InitFluidData();

		m_Factor.resize(m_ParticleCount, 0.0f);
		m_Kappa.resize(m_ParticleCount, 0.0f);
		m_KappaVelocity.resize(m_ParticleCount, 0.0f);
		m_DensityAdvection.resize(m_ParticleCount, 0.0f);

		{
			StaticRigidBodyDescription rigidBodyDesc;
			rigidBodyDesc.SourceMesh = "Resources/Models/Cube.obj";
			rigidBodyDesc.Position = { 0.0f, 3.0f, 0.0f };
			rigidBodyDesc.Rotation = glm::angleAxis(glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 0.0f));
			rigidBodyDesc.Scale = { 2.0f, 0.25f, 2.0f };
			rigidBodyDesc.Inverted = false;
			rigidBodyDesc.Padding = 0.0f;
			rigidBodyDesc.CollisionMapResolution = { 20, 20, 20 };

			StaticRigidBody* rigidBody = new StaticRigidBody(rigidBodyDesc, this);
			m_RigidBodies.push_back(rigidBody);
		}

		{
			StaticRigidBodyDescription rigidBodyDesc;
			rigidBodyDesc.SourceMesh = "Resources/Models/Torus.obj";
			rigidBodyDesc.Position = { 0.0f, 4.0f, 0.0f };
			rigidBodyDesc.Rotation = glm::angleAxis(glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 0.0f));
			rigidBodyDesc.Scale = { 0.5f, 0.5f, 0.5f };
			rigidBodyDesc.Inverted = false;
			rigidBodyDesc.Padding = 0.0f;
			rigidBodyDesc.CollisionMapResolution = { 20, 20, 20 };

			StaticRigidBody* rigidBody = new StaticRigidBody(rigidBodyDesc, this);
			m_RigidBodies.push_back(rigidBody);
		}

		m_SurfaceTensionSolver = new SurfaceTensionSolverDFSPH(this);
		m_ViscositySolver = new ViscositySolverDFSPH(this);
	}

	DFSPHSimulation::~DFSPHSimulation()
	{ }

	void DFSPHSimulation::OnUpdate()
	{
		if (paused) {
			return;
		}

		if (m_FrameCounter % 500 == 0) {
			m_NeighborhoodSearch->ZSort();

			if (m_ParticleCount > 0) {
				const PointSet& pointSet = m_NeighborhoodSearch->GetPointSet(0);

				pointSet.SortField(&m_ParticlePositions[0]);
				pointSet.SortField(&m_ParticleVelocities[0]);
				pointSet.SortField(&m_ParticleAccelerations[0]);
				pointSet.SortField(&m_ParticleMasses[0]);
				pointSet.SortField(&m_ParticleDensities[0]);
				pointSet.SortField(&m_Kappa[0]);
				pointSet.SortField(&m_KappaVelocity[0]);

				m_ViscositySolver->Sort(pointSet);
				m_SurfaceTensionSolver->Sort(pointSet);
			}
		}

		m_FrameCounter++;
		m_NeighborhoodSearch->FindNeighbors();

		PrecomputeValues();
		ComputeVolumeAndBoundaryX();
		ComputeDensities();
		ComputeDFSPHFactor();

		if (m_Description.EnableDivergenceSolver) {
			DivergenceSolve();
		}

		ClearAccelerations();

		// Non-Pressure forces
		m_SurfaceTensionSolver->OnUpdate();
		m_ViscositySolver->OnUpdate();

		UpdateTimeStepSize();

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)m_ParticleCount; i++) {
				m_ParticleVelocities[i] += m_TimeStepSize * m_ParticleAccelerations[i];
			}
		}

		PressureSolve();

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)m_ParticleCount; i++)
			{
				m_ParticlePositions[i] += m_TimeStepSize * m_ParticleVelocities[i];
			}
		}
	}

	void DFSPHSimulation::OnRenderTemp()
	{
		for (unsigned int i = 0; i < m_RigidBodies.size(); i++)
		{
			const StaticRigidBodyDescription& desc = m_RigidBodies[i]->GetDescription();

			if (desc.Inverted == false) {
				glm::mat4 t = glm::translate(glm::mat4(1.0f), desc.Position);
				// t = glm::scale(t, { 1.0f, 1.0f, 1.0f });
				t = t * glm::toMat4(desc.Rotation);
				m_Material->Set("model", t);
				Renderer::DrawTriangles(m_RigidBodies[i]->GetGeometry().GetVAO(), m_RigidBodies[i]->GetGeometry().GetVertexCount(), m_Material);
			}
		}
	
		for (size_t i = 0; i < m_ParticleCount; i++)
		{
			// default
			Renderer::DrawPoint(m_ParticlePositions[i], { 0.65f, 0.65f, 0.65f, 1 }, m_Description.ParticleRadius * 32);

			// density
			// float v = m_ParticleDensities[i] / 1000.0f;
			// Renderer::DrawPoint(m_ParticlePositions[i], { v, 0, v, 1}, m_Description.ParticleRadius * 32);

			// factor
			// float v = 1.0f + m_Factor[i] * 500;
			// Renderer::DrawPoint(m_ParticlePositions[i], { v, 0, v, 1 }, m_Description.ParticleRadius * 32);

			// Kappa
			// float v = -m_Kappa[i] * 10000;
			// Renderer::DrawPoint(m_ParticlePositions[i], { v, 0, v, 1 }, m_Description.ParticleRadius * 32);

			// Kappa V
			// float v = -m_KappaVelocity[i] * 10;
			// Renderer::DrawPoint(m_ParticlePositions[i], { v, 0, v, 1 }, m_Description.ParticleRadius * 32);

			// Classifier output
			// float v = m_SurfaceTensionSolver->GetClassifierOutput(i);
			// Renderer::DrawPoint(m_ParticlePositions[i], { v, 0, v, 1 }, m_Description.ParticleRadius * 32);
		}
	}

	void DFSPHSimulation::SetParticleRadius(float value)
	{
		m_SupportRadius = static_cast<float>(4.0) * value;
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
				const glm::vec3& particlePosition = m_ParticlePositions[i];
				for (unsigned int j = 0; j < m_RigidBodies.size(); j++)
				{
					StaticRigidBody* rigidBody = m_RigidBodies[j];
					glm::vec3& rigidBodyXJ = rigidBody->GetBoundaryXJ(i);
					float& rigidBodyVolume = rigidBody->GetBoundaryVolume(i);

					rigidBodyVolume = 0.0f;
					rigidBodyXJ = { 0.0f, 0.0f, 0.0f };

					const glm::mat3 rigidBodyRotation = glm::toMat3(rigidBody->GetRotation());

					glm::dvec3 normal;
					const glm::dvec3 localPosition = (glm::transpose(rigidBodyRotation) * particlePosition);

					std::array<unsigned int, 32> cell;
					glm::dvec3 c0;
					std::array<double, 32> N;
					std::array<std::array<double, 3>, 32> dN;
					double dist = std::numeric_limits<double>::max();

					if (rigidBody->GetCollisionMap()->DetermineShapeFunctions(0, localPosition, cell, c0, N, &dN)) {
						dist = rigidBody->GetCollisionMap()->Interpolate(0, localPosition, cell, c0, N, &normal, &dN);
					}

					if ((dist > 0.0) && (static_cast<float>(dist) < m_SupportRadius))
					{
						const double volume = rigidBody->GetCollisionMap()->Interpolate(1, localPosition, cell, c0, N);
						if ((volume > 0.0) && (volume != std::numeric_limits<double>::max()))
						{
							rigidBodyVolume = static_cast<float>(volume);
							normal = rigidBodyRotation * normal;
							const double normalLength = glm::length(normal);

							if (normalLength > 1.0e-9)
							{
								normal /= normalLength;
								const float d = std::max((static_cast<float>(dist) + static_cast<float>(0.5) * m_Description.ParticleRadius), static_cast<float>(2.0) * m_Description.ParticleRadius);
								rigidBodyXJ = (particlePosition - d * (glm::vec3)normal);
							}
							else
							{
								rigidBodyVolume = 0.0f;
							}
						}
						else
						{
							rigidBodyVolume = 0.0f;
						}
					}
					else if (dist <= 0.0)
					{
						if (dist != std::numeric_limits<double>::max())
						{
							normal = rigidBodyRotation * normal;
							const double normalLength = glm::length(normal);

							if (normalLength > 1.0e-5)
							{
								normal /= normalLength;
								// Project to surface
								float delta = static_cast<float>(2.0) * m_Description.ParticleRadius - static_cast<float>(dist);
								delta = std::min(delta, static_cast<float>(0.1) * m_Description.ParticleRadius);
								m_ParticlePositions[i] = (particlePosition + delta * (glm::vec3)normal);
								m_ParticleVelocities[i] = { 0.0f, 0.0f, 0.0f };
							}
						}

						rigidBodyVolume = 0.0f;
					}
					else
					{
						rigidBodyVolume = 0.0f;
					}
				}
			}
		}
	}

	void DFSPHSimulation::ComputeDensities()
	{
		auto* m_Base = this; // TEMP

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < (int)m_ParticleCount; i++)
			{
				const glm::vec3& xi = m_ParticlePositions[i];
				float& density = m_ParticleDensities[i];
				density = m_Volume * CubicKernelAVX::WZero();

				Scalar8 density_avx(0.0f);
				Scalar3f8 xi_avx(xi);

				ForAllFluidNeighborsAVX(
					const Scalar8 Vj_avx = ConvertZero(m_Volume, count);
					density_avx += Vj_avx * CubicKernelAVX::W(xi_avx - xj_avx);
				);

				density += density_avx.Reduce();
				ForAllVolumeMaps(
					density += Vj * PrecomputedCubicKernel::W(xi - xj);
				);

				density *= m_Density0;
			}
		}
	}

	void DFSPHSimulation::ComputeDFSPHFactor()
	{
		auto* m_Base = this;

		#pragma omp parallel default(shared)
		{
			#pragma omp for schedule(static)  
			for (int i = 0; i < m_ParticleCount; i++)
			{
				float sumGradientPK;
				glm::vec3 gradientPI;
				Scalar3f8 positionAVX(m_ParticlePositions[i]);
				Scalar8 sumGradientPKAVX(0.0f);
				Scalar3f8 gradientPIAVX;
				gradientPIAVX.SetZero();

				ForAllFluidNeighborsAVXNOX(
					ComputeVJGradientW();
					const Scalar3f8 & gradC_j = V_gradW;
					sumGradientPKAVX += gradC_j.SquaredNorm();
					gradientPIAVX = gradientPIAVX + gradC_j;
				);

				sumGradientPK = sumGradientPKAVX.Reduce();
				gradientPI[0] = gradientPIAVX.x().Reduce();
				gradientPI[1] = gradientPIAVX.y().Reduce();
				gradientPI[2] = gradientPIAVX.z().Reduce();

				ForAllVolumeMaps(
					const glm::vec3 gradientPJ = -Vj * PrecomputedCubicKernel::GradientW(m_ParticlePositions[i] - xj);
					gradientPI -= gradientPJ;
				);

				sumGradientPK += glm::length2(gradientPI);
				m_Factor[i] = sumGradientPK > EPS ? -static_cast<float>(1.0) / (sumGradientPK) : 0.0f;
			}
		}
	}

	void DFSPHSimulation::DivergenceSolve()
	{
		const float h = m_TimeStepSize;
		const float invH = static_cast<float>(1.0) / h;
		const unsigned int maxIter = m_Description.MaxVolumeSolverIterations;
		const float maxError = m_Description.MaxVolumeError;

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
			m_KappaVelocity[i] *= h;


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
					m_KappaVelocity[i] = static_cast<float>(0.5) * std::max(m_KappaVelocity[i], static_cast<float>(-0.5)) * invH;
				else
					m_KappaVelocity[i] = 0.0;
			}

#pragma omp for schedule(static)  
			for (int i = 0; i < (int)numParticles; i++)
			{
				{
					const float ki = m_KappaVelocity[i];
					const glm::vec3& xi = m_ParticlePositions[i];
					glm::vec3& vi = m_ParticleVelocities[i];

					Scalar8 ki_avx(ki);
					Scalar3f8 xi_avx(xi);
					Scalar3f8 delta_vi;
					delta_vi.SetZero();

					ForAllFluidNeighborsAVXNOX(
						ComputeVJGradientW();
						const Scalar8 densityFrac_avx(m_Density0 / density0);
						const Scalar8 kj_avx = ConvertZero(&GetNeighborList(0, 0, i)[j], &m_KappaVelocity[0], count);
						const Scalar8 kSum_avx = ki_avx + densityFrac_avx * kj_avx;

						delta_vi = delta_vi + (V_gradW * (h_avx * kSum_avx));
					)

						vi[0] += delta_vi.x().Reduce();
					vi[1] += delta_vi.y().Reduce();
					vi[2] += delta_vi.z().Reduce();

					if (fabs(ki) > EPS)
					{
						ForAllVolumeMaps(
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

		ForAllFluidNeighborsAVXNOX(
			ComputeVJGradientW();
		const Scalar3f8 vj_avx = ConvertScalarZero(&GetNeighborList(0, 0, i)[j], &m_ParticleVelocities[0], count);
		densityAdv_avx += (vi_avx - vj_avx).Dot(V_gradW);
		);

		float& densityAdv = m_DensityAdvection[i];
		densityAdv = densityAdv_avx.Reduce();

		ForAllVolumeMaps(
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

				m_KappaVelocity[i] += ki;

				glm::vec3& vi = m_ParticleVelocities[i];
				const glm::vec3& xi = m_ParticlePositions[i];

				Scalar8 ki_avx(ki);
				Scalar3f8 xi_avx(xi);
				Scalar3f8 delta_vi;
				delta_vi.SetZero();

				ForAllFluidNeighborsAVXNOX(
					ComputeVJGradientW();
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
					ForAllVolumeMaps(
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
				a = m_Description.Gravity;
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
		const float radius = m_Description.ParticleRadius;
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
		h = static_cast<float>(0.4) * (diameter / (sqrt(maxVel)));

		h = std::min(h, m_Description.CFLMaxTimeStepSize);
		h = std::max(h, m_Description.CFLMinTimeStepSize);

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


		while ((!chk || (m_PressureSolverIterations < m_Description.MinPressureSolverIteratations)) && (m_PressureSolverIterations < m_Description.MaxPressureSolverIterations)) {
			chk = true;

			avg_density_err = 0.0;
			PressureSolveIteration(avg_density_err);

			// Maximal allowed density fluctuation
			const float eta = m_Description.MaxPressureSolverError * static_cast<float>(0.01) * density0;  // maxError is given in percent
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

				ForAllFluidNeighborsAVXNOX(
					ComputeVJGradientW();
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
					ForAllVolumeMaps(
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

		ForAllFluidNeighborsAVXNOX(
			ComputeVJGradientW();
		const Scalar3f8 vj_avx = ConvertScalarZero(&GetNeighborList(0, 0, i)[j], &m_ParticleVelocities[0], count);
		delta_avx += (vi_avx - vj_avx).Dot(V_gradW);
		);

		delta = delta_avx.Reduce();

		ForAllVolumeMaps(
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

				ForAllFluidNeighborsAVXNOX(
					ComputeVJGradientW();
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
					ForAllVolumeMaps(
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

				ForAllFluidNeighborsAVX(
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
		float diam = static_cast<float>(2.0) * m_Description.ParticleRadius;
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

	ViscositySolverDFSPH::ViscositySolverDFSPH(DFSPHSimulation* base)
	{
		m_MaxIterations = 100;
		m_BoundaryViscosity = 1;
		m_Viscosity = 1;
		m_TangentialDistanceFactor = static_cast<float>(0.5);

		m_ViscosityDifference.resize(base->GetParticleCount(), glm::vec3(0.0, 0.0, 0.0));
		m_Base = base;
	}

	void ViscositySolverDFSPH::DiagonalMatrixElement(const unsigned int i, glm::mat3x3& result, void* userData, DFSPHSimulation* m_Base)
	{
		ViscositySolverDFSPH* visco = (ViscositySolverDFSPH*)userData;
		auto* sim = m_Base;

		const float density0 = sim->GetDensity0();
		const float d = 10.0;

		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float mu = visco->m_Viscosity * density0;
		const float mub = visco->m_BoundaryViscosity * density0;
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

		ForAllFluidNeighborsInSamePhaseAVX(
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

			ForAllVolumeMaps(
				const glm::vec3 xixj = xi - xj;
				glm::vec3 normal = -xixj;
				const float normalLength = std::sqrt(glm::dot(normal, normal));

				if (normalLength > static_cast<float>(0.0001))
				{
					normal /= normalLength;

					glm::vec3 t1;
					glm::vec3 t2;
					GetOrthogonalVectors(normal, t1, t2);

					const float dist = visco->m_TangentialDistanceFactor * h;
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

	void ViscositySolverDFSPH::ComputeRHS(std::vector<float>& b, std::vector<float>& g)
	{
		const int numParticles = (int)m_Base->GetParticleCount();
		auto* sim = m_Base;
		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float density0 = sim->GetDensity0();
		const float mu = m_Viscosity * density0;
		const float mub = m_BoundaryViscosity * density0;
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

					ForAllVolumeMaps(
						const glm::vec3 xixj = xi - xj;
						glm::vec3 normal = -xixj;
						const float normalLength = std::sqrt(glm::dot(normal, normal));

						if (normalLength > static_cast<float>(0.0001)) {
							normal /= normalLength;

							glm::vec3 t1;
							glm::vec3 t2;
							GetOrthogonalVectors(normal, t1, t2);

							const float dist = m_TangentialDistanceFactor * h;
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

				g[3 * i + 0] = vi[0] + m_ViscosityDifference[i][0];
				g[3 * i + 1] = vi[1] + m_ViscosityDifference[i][1];
				g[3 * i + 2] = vi[2] + m_ViscosityDifference[i][2];
			}
		}
	}

	void ViscositySolverDFSPH::ApplyForces(const std::vector<float>& x)
	{
		const int numParticles = (int)m_Base->GetParticleCount();
		auto* sim = m_Base;
		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float density0 = sim->GetDensity0();
		const float mu = m_Viscosity * density0;
		const float mub = m_BoundaryViscosity * density0;
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
				m_ViscosityDifference[i] = (newVi - sim->GetParticleVelocity(i));

				const glm::vec3& xi = sim->GetParticlePosition(i);
				const float density_i = sim->GetParticleDensity(i);
				const float m_i = sim->GetParticleMass(i);

				if (mub != 0.0)
				{
					const auto& m_RigidBodies = m_Base->GetRigidBodies();
					ForAllVolumeMaps(
						const glm::vec3 xixj = xi - xj;
					glm::vec3 normal = -xixj;
					const float normalLength = std::sqrt(glm::dot(normal, normal));

					if (normalLength > static_cast<float>(0.0001)) {
						normal /= normalLength;

						glm::vec3 t1;
						glm::vec3 t2;
						GetOrthogonalVectors(normal, t1, t2);

						const float dist = m_TangentialDistanceFactor * h;
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

	void ViscositySolverDFSPH::OnUpdate() {
		const unsigned int numParticles = (int)m_Base->GetParticleCount();
		if (numParticles == 0) {
			return;
		}

		const float density0 = m_Base->GetDensity0();
		const float h = m_Base->GetTimeStepSize();

		MatrixReplacement A(3 * numParticles, MatrixVectorProduct, (void*)this, m_Base);
		m_Solver.GetPreconditioner().Init(numParticles, DiagonalMatrixElement, (void*)this, m_Base);

		m_Solver.m_tolerance = m_MaxError;
		m_Solver.m_MaxPressureSolverIterations = m_MaxIterations;
		m_Solver.Compute(A);

		std::vector<float> b(3 * numParticles);
		std::vector<float> g(3 * numParticles);
		std::vector<float> x(3 * numParticles);

		ComputeRHS(b, g);
		m_Solver.SolveWithGuess(b, g, x);
		ApplyForces(x);
	}

	void ViscositySolverDFSPH::Sort(const PointSet& pointSet)
	{
		pointSet.SortField(&m_ViscosityDifference[0]);
	}

	void ViscositySolverDFSPH::MatrixVectorProduct(const std::vector<float>& vec, std::vector<float>& result, void* userData, DFSPHSimulation* m_Base)
	{
		ViscositySolverDFSPH* visco = (ViscositySolverDFSPH*)userData;
		auto* sim = m_Base;
		const unsigned int numParticles = sim->GetParticleCount();

		const float h = sim->GetParticleSupportRadius();
		const float h2 = h * h;
		const float dt = sim->GetTimeStepSize();
		const float density0 = sim->GetDensity0();
		const float mu = visco->m_Viscosity * density0;
		const float mub = visco->m_BoundaryViscosity * density0;
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

				ForAllFluidNeighborsInSamePhaseAVX(
					ComputeVJGradientSamePhase();
					const Scalar8 density_j_avx = ConvertOne(&sim->GetNeighborList(0, 0, i)[j], &sim->GetParticleDensity(0), count);
					const Scalar3f8 xixj = xi_avx - xj_avx;
					const Scalar3f8 vj_avx = ConvertScalarZero(&sim->GetNeighborList(0, 0, i)[j], &vec[0], count);

					delta_ai_avx = delta_ai_avx + (V_gradW * ((d_mu_rho0 / density_j_avx) * (vi_avx - vj_avx).Dot(xixj) / (xixj.SquaredNorm() + h2_001)));
				);

				if (mub != 0.0)
				{
					const auto& m_RigidBodies = m_Base->GetRigidBodies();

					ForAllVolumeMaps(
						const glm::vec3 xixj = xi - xj;
						glm::vec3 normal = -xixj;
						const float normalLength = std::sqrt(glm::dot(normal, normal));
						if (normalLength > static_cast<float>(0.0001)) {
							normal /= normalLength;

							glm::vec3 t1;
							glm::vec3 t2;
							GetOrthogonalVectors(normal, t1, t2);

							const float dist = visco->m_TangentialDistanceFactor * h;
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

	SurfaceTensionSolverDFSPH::SurfaceTensionSolverDFSPH(DFSPHSimulation* base)
		:
		m_SurfaceTension(.3f)
		, m_SamplesPerSecond(10000) // 10000 // 36000 // 48000 // 60000
		, m_SmoothingFactor(0.5)
		, m_Factor(0.8f)
		, m_ParticleRadius(base->GetParticleSupportRadius())
		, m_NeighborParticleRadius(m_Factor* m_ParticleRadius)
		, m_ClassifierSlope(74.688796680497925f)
		, m_ClassifierConstant(28)
		, m_TemporalSmoothing(false)
		, m_SamplesPerStep(-1)
		, m_ClassifierOffset(2)
		, m_PCANMix(0.75f)
		, m_PCACMix(0.5f)
		, m_CurvatureLimit(16)
		, m_SmoothPassCount(1)
	{
		m_Base = base;
		m_MonteCarloSurfaceNormals          .resize(base->GetParticleCount(), { 0.0f, 0.0f, 0.0f });
		m_MonteCarloSurfaceNormalsSmooth   .resize(base->GetParticleCount(), { 0.0f, 0.0f, 0.0f });
		m_FinalCurvature    .resize(base->GetParticleCount(),   0.0f);
		m_SmoothedCurvature     .resize(base->GetParticleCount(),   0.0f);
		m_MonteCarloSurfaceCurvature             .resize(base->GetParticleCount(),   0.0f);
		m_MonteCarloSurfaceCurvatureSmooth      .resize(base->GetParticleCount(),   0.0f);
		m_DeltaFinalCurvature.resize(base->GetParticleCount(),   0.0f);
		m_ClassifierInput    .resize(base->GetParticleCount(),   0.0f);
		m_ClassifierOutput   .resize(base->GetParticleCount(),   0.0f);
	}

	void SurfaceTensionSolverDFSPH::OnUpdate()
	{
		float timeStep = m_Base->GetTimeStepSize();

		m_NeighborParticleRadius = m_ParticleRadius * m_Factor;

		auto* sim = m_Base;

		const float supportRadius = sim->GetParticleSupportRadius();
		const unsigned int numParticles = sim->GetParticleCount();
		const float k = m_SurfaceTension;

		unsigned int NrOfSamples;

		if (m_SamplesPerStep > 0)
			NrOfSamples = m_SamplesPerStep;
		else
			NrOfSamples = int(m_SamplesPerSecond * timeStep);

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
				m_MonteCarloSurfaceNormals[i] = glm::vec3(0,  0, 0);
				m_MonteCarloSurfaceNormalsSmooth[i] = glm::vec3(0,  0, 0);

				m_MonteCarloSurfaceCurvature[i] = 0.0;
				m_MonteCarloSurfaceCurvatureSmooth[i] = 0.0;
				m_SmoothedCurvature[i] = 0.0;
				m_FinalCurvature[i] = 0.0;


				// -- compute center of mass of current particle

				glm::vec3 centerofMasses = glm::vec3(0,  0, 0);
				int numberOfNeighbours = sim->NumberOfNeighbors(0, 0, i);

				if (numberOfNeighbours == 0)
				{
					m_MonteCarloSurfaceCurvature[i] = static_cast<float>(1.0) / supportRadius;
					continue;
				}

				const glm::vec3& xi = sim->GetParticlePosition(i);

				ForAllFluidNeighborsInSamePhase(
					glm::vec3 xjxi = (xj - xi);
					centerofMasses += xjxi;
				);

				centerofMasses /= supportRadius;

				// cache classifier input, could also be recomputed later to avoid caching
				m_ClassifierInput[i] = glm::length(centerofMasses) / static_cast<float>(numberOfNeighbours);


				// -- if it is a surface classified particle
				if (ClassifyParticleConfigurable(m_ClassifierInput[i], numberOfNeighbours)) //EvaluateNetwork also possible
				{

					// -- create monte carlo samples on particle
					std::vector<glm::vec3> points = GetSphereSamplesLookUp(
						NrOfSamples, supportRadius, i * NrOfSamples, haltonVec323, static_cast<int>(haltonVec323.size())); // 8.5 // 15.0(double) // 9.0(float)

					//  -- remove samples covered by neighbor spheres
					ForAllFluidNeighborsInSamePhase(
						glm::vec3 xjxi = (xj - xi);
						for (int p = static_cast<int>(points.size()) - 1; p >= 0; --p)
						{
							glm::vec3 vec = (points[p] - xjxi);
							float dist = glm::length2(vec);

							if (dist <= pow((m_NeighborParticleRadius / m_ParticleRadius), 2) * supportRadius * supportRadius) {
								points.erase(points.begin() + p);
								erased++;
							}
						})

						// -- estimate normal by left over sample directions
						for (int p = static_cast<int>(points.size()) - 1; p >= 0; --p)
							m_MonteCarloSurfaceNormals[i] += points[p];


					// -- if surface classified and non-overlapping neighborhood spheres
					if (points.size() > 0)
					{
						m_MonteCarloSurfaceNormals[i] = glm::normalize(m_MonteCarloSurfaceNormals[i]);

						// -- estimate curvature by sample ratio and particle radii
						m_MonteCarloSurfaceCurvature[i] = (static_cast<float>(1.0) / supportRadius) * static_cast<float>(-2.0) * pow((static_cast<float>(1.0) - (m_NeighborParticleRadius * m_NeighborParticleRadius / (m_ParticleRadius * m_ParticleRadius))), static_cast<float>(-0.5)) *
							cos(acos(static_cast<float>(1.0) - static_cast<float>(2.0) * (static_cast<float>(points.size()) / static_cast<float>(NrOfSamples))) + asin(m_NeighborParticleRadius / m_ParticleRadius));

						m_ClassifierOutput[i] = 1.0; // -- used to visualize surface points (blue in the paper)
					}
					else
					{
						// -- correct false positives to inner points
						m_MonteCarloSurfaceNormals[i] = glm::vec3(0,  0, 0);
						m_MonteCarloSurfaceCurvature[i] = 0.0;
						m_ClassifierOutput[i] = 0.5; // -- used for visualize post-correction points (white in the paper)
					}
				}
				else
				{
					// -- used to visualize inner points (green in the paper)
					m_ClassifierOutput[i] = 0.0;
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
				if (m_MonteCarloSurfaceNormals[i] != glm::vec3(0,  0, 0))
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

					ForAllFluidNeighborsInSamePhase(
						if (m_MonteCarloSurfaceNormals[neighborIndex] != glm::vec3(0,  0, 0))
						{
							glm::vec3& xj = sim->GetParticlePosition(neighborIndex);
							glm::vec3 xjxi = (xj - xi);

							surfCentDir += xjxi;
							centroid += xj;
							t_count++;

							float distanceji = glm::length(xjxi);

							normalCorrection += m_MonteCarloSurfaceNormals[neighborIndex] * (1 - distanceji / supportRadius);
							correctionForCurvature += m_MonteCarloSurfaceCurvature[neighborIndex] * (1 - distanceji / supportRadius);
							correctionFactor += (1 - distanceji / supportRadius);
						}
						)

						normalCorrection = glm::normalize(normalCorrection);
						m_MonteCarloSurfaceNormalsSmooth[i] = (1 - m_SmoothingFactor) * m_MonteCarloSurfaceNormals[i] + m_SmoothingFactor * normalCorrection;
						m_MonteCarloSurfaceNormalsSmooth[i] = glm::normalize(m_MonteCarloSurfaceNormalsSmooth[i]);

						m_MonteCarloSurfaceCurvatureSmooth[i] =
							((static_cast<float>(1.0) - m_SmoothingFactor) * m_MonteCarloSurfaceCurvature[i] + m_SmoothingFactor * correctionForCurvature) /
							((static_cast<float>(1.0) - m_SmoothingFactor) + m_SmoothingFactor * correctionFactor);
				}
			}
		}


		// ################################################################################################
		// ## third pass, final blending and temporal smoothing
		// ################################################################################################

		m_SmoothPassCount = std::max(1, m_SmoothPassCount);

		for (int si = 0; si < m_SmoothPassCount; si++)
		{
			// smoothing pass 2 for sphericity
#pragma omp parallel default(shared)
			{
#pragma omp for schedule(static)  
				for (int i = 0; i < (int)numParticles; i++)
				{
					if (m_MonteCarloSurfaceNormals[i] != glm::vec3(0, 0, 0))
					{
						int count = 0;
						float CsCorr = 0.0;

						const glm::vec3& xi = sim->GetParticlePosition(i);

						ForAllFluidNeighborsInSamePhase(
							if (m_MonteCarloSurfaceNormals[neighborIndex] != glm::vec3(0, 0, 0))
							{
								count++;
							})


							if (count > 0)
								m_SmoothedCurvature[i] = static_cast<float>(0.25) * m_SmoothedCurvature[i] + static_cast<float>(0.75) * CsCorr / static_cast<float>(count);

							m_SmoothedCurvature[i] /= supportRadius;
							m_SmoothedCurvature[i] *= 20.0;

							if (m_SmoothedCurvature[i] > 0.0)
								m_SmoothedCurvature[i] = std::min(0.5f / supportRadius, m_SmoothedCurvature[i]);
							else
								m_SmoothedCurvature[i] = std::max(-0.5f / supportRadius, m_SmoothedCurvature[i]);


							glm::vec3 final_normal = glm::vec3(0, 0, 0);
							float     final_curvature = m_MonteCarloSurfaceCurvatureSmooth[i];

							final_normal = m_MonteCarloSurfaceNormalsSmooth[i];
							final_curvature = m_MonteCarloSurfaceCurvatureSmooth[i];

							if (m_TemporalSmoothing)
								m_FinalCurvature[i] = static_cast<float>(0.05) * final_curvature + static_cast<float>(0.95) * m_DeltaFinalCurvature[i];
							else
								m_FinalCurvature[i] = final_curvature;

							glm::vec3 force = final_normal * k * m_FinalCurvature[i];

							glm::vec3& ai = sim->GetParticleAcceleration(i);
							ai -= (1 / sim->GetParticleMass(i)) * force;

							m_DeltaFinalCurvature[i] = m_FinalCurvature[i];
					}
					else // non surface particle blend 0.0 curvature
					{

						if (m_TemporalSmoothing)
							m_FinalCurvature[i] = static_cast<float>(0.95) * m_DeltaFinalCurvature[i];
						else
							m_FinalCurvature[i] = 0.0;

						m_DeltaFinalCurvature[i] = m_FinalCurvature[i];
					}
				}
			}
		}
	}

	void SurfaceTensionSolverDFSPH::Sort(const PointSet& pointSet)
	{
		pointSet.SortField(&m_MonteCarloSurfaceNormals[0]);
		pointSet.SortField(&m_FinalCurvature[0]);
		pointSet.SortField(&m_SmoothedCurvature[0]);
		pointSet.SortField(&m_MonteCarloSurfaceCurvature[0]);
		pointSet.SortField(&m_MonteCarloSurfaceCurvatureSmooth[0]);
		pointSet.SortField(&m_MonteCarloSurfaceNormalsSmooth[0]);
		pointSet.SortField(&m_DeltaFinalCurvature[0]);
		pointSet.SortField(&m_ClassifierInput[0]);
		pointSet.SortField(&m_ClassifierOutput[0]);
	}

	bool SurfaceTensionSolverDFSPH::ClassifyParticleConfigurable(double com, int non, double offset)
	{
		double neighborsOnTheLine = m_ClassifierSlope * com + m_ClassifierConstant + offset;

		if (non <= neighborsOnTheLine) {
			return true;
		}
		else {
			return false;
		}
	}

	std::vector<glm::vec3> SurfaceTensionSolverDFSPH::GetSphereSamplesLookUp(int N, float supportRadius, int start, const std::vector<float>& vec3, int mod)
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
