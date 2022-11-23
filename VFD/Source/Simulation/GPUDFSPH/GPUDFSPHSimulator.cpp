#include "pch.h"
#include "GPUDFSPHSimulator.h"
#include <Core/Structures/BoundingBox.h>
#include "Renderer/Renderer.h"

namespace vfd
{
	const float radius = 0.01f;

	GPUDFSPHSimulation::GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc)
		: m_Description(desc)
	{
		Ref<TriangleMesh> mesh = Ref<TriangleMesh>::Create("Resources/Models/Bunny.obj");
		m_SDF = Ref<GPUSDF>::Create(mesh);

		BoundingBox<glm::vec3> bounds = m_SDF->GetDomain();

		bounds.max = bounds.max * 2.0f;
		bounds.min = bounds.min * 2.0f;

		const float diameter = 2.0f * radius;

		uint32_t currentSample = 0;
		uint32_t counterX = 0;
		uint32_t counterY = 0;

		float shiftX = diameter;
		float shiftY = diameter;

		shiftX = std::sqrtf(3.0f) * radius;
		shiftY = std::sqrtf(6.0f) * diameter / 3.0f;

		for (float z = bounds.min.z; z <= bounds.max.z; z += diameter) {
			for (float y = bounds.min.y; y <= bounds.max.y; y += shiftY) {
				for (float x = bounds.min.x; x <= bounds.max.x; x += shiftX) {
					glm::vec3 particlePosition;

					particlePosition = glm::vec3(x, y + radius, z + radius);
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

					particlePosition += shift;

					// Check if the current sample is inside the model
					if (m_SDF->GetDistanceTricubic(particlePosition) < 0.0f) {
						samples.push_back(particlePosition);
					}

					currentSample++;
					counterX++;
				}
				counterX = 0;
				counterY++;
			}
			counterY = 0;
		}
	}

	void GPUDFSPHSimulation::OnRender()
	{
		for (size_t i = 0; i < samples.size(); i++)
		{
			Renderer::DrawPoint(samples[i], { .5, .5, .5, 1.0f }, radius * 32);
		}
	}

	void GPUDFSPHSimulation::OnUpdate()
	{

	}
}