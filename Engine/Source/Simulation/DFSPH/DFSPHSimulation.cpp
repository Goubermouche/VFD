#include "pch.h"
#include "DFSPHSimulation.h"

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
			data->meshFile = "Resources/Models/SDFSafe/UnitBox.obj";
			data->translation = { 0, -0.25, 0 };
			data->rotation = glm::mat4(1.0f);
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
	}

	DFSPHSimulation::~DFSPHSimulation()
	{
	}

	void DFSPHSimulation::OnUpdate()
	{
	}
}
