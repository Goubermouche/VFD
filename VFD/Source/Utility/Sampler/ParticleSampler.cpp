#include "pch.h"
#include "ParticleSampler.h"

#include "Simulation/DFSPH/DensityMap/DensityMap.cuh"

namespace vfd {
	std::vector<glm::vec3> ParticleSampler::SampleMeshVolume(const Ref<EdgeMesh>& mesh, const float radius, const glm::uvec3& resolution, const bool inverted, const SampleMode sampleMode)
	{
		ERR("callllll")
		BoundingBox bounds(mesh->GetVertices());

		std::cout << bounds.min.x << " " << bounds.min.y << " " << bounds.min.z << '\n';
		std::cout << bounds.max.x << " " << bounds.max.y << " " << bounds.max.z << '\n';
		std::cout << "--------\n";

		Ref<DensityMap> sdf = Ref<DensityMap>::Create(mesh, bounds, resolution, inverted);

		const float diameter = 2.0f * radius;

		uint32_t currentSample = 0;
		uint32_t counterX = 0;
		uint32_t counterY = 0;

		float shiftX = diameter;
		float shiftY = diameter;

		std::vector<glm::vec3> samples;

		if (sampleMode == SampleMode::MediumDensity) {
			shiftY = std::sqrtf(3.0f) * radius;
		}
		else if (sampleMode == SampleMode::MaxDensity)
		{
			shiftX = std::sqrtf(3.0f) * radius;
			shiftY = std::sqrtf(6.0f) * diameter / 3.0f;
		}

		for (float z = bounds.min.z; z <= bounds.max.z; z += diameter) {
			for (float y = bounds.min.y; y <= bounds.max.y; y += shiftY) {
				for (float x = bounds.min.x; x <= bounds.max.x; x += shiftX) {
					glm::vec3 particlePosition;

					switch (sampleMode)
					{
					case SampleMode::MinDensity:
					{
						particlePosition = { x + radius, y + radius, z + radius };
						break;
					}
					case SampleMode::MediumDensity:
					{
						if (counterY % 2 == 0) {
							particlePosition = glm::vec3(x, y + radius, z + radius);
						}
						else {
							particlePosition = glm::vec3(x + radius, y + radius, z);
						}
						break;
					}
					case SampleMode::MaxDensity:
					{
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
						break;
					}
					}

					// Check of the current sample is inside the model
					if (sdf->GetDistance(particlePosition, 0.0f) < 0.0f) {
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

		return samples;
	}
}