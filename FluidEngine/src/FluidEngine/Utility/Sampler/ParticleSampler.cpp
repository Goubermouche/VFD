#include "pch.h"
#include "ParticleSampler.h"

#include "FluidEngine/Utility/SDF/SDF.h"

namespace fe {
	std::vector<glm::vec3> ParticleSampler::SampleMeshVolume(const EdgeMesh& mesh, const float radius, const glm::ivec3& resolution, const bool inverted, const SampleMode sampleMode)
	{
		BoundingBox bounds = BoundingBox::ComputeBoundingBox(mesh.GetVertices());
		Ref<SDF> sdf = Ref<SDF>::Create(mesh, bounds, resolution, inverted);

		const float diameter = 2.0f * radius;

		// sample object
		const uint32_t numberOfSamplePoints =
			(((uint32_t)((1.0f / diameter) * (bounds.max.z - bounds.min.z))) + 1) *
			(((uint32_t)((1.0f / diameter) * (bounds.max.y - bounds.min.y))) + 1) *
			(((uint32_t)((1.0f / diameter) * (bounds.max.x - bounds.min.x))) + 1);

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

					// sample is inside the model
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