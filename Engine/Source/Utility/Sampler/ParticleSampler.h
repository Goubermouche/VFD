#ifndef VOLUME_SAMPLER_H_
#define VOLUME_SAMPLER_H_

#include "Renderer/Mesh/EdgeMesh.h"

namespace fe {
	enum class SampleMode {
		MinDensity = 0,
		MediumDensity = 1, 
		MaxDensity = 2
	};

	class ParticleSampler
	{
	public:
		static std::vector<glm::vec3> SampleMeshVolume(const EdgeMesh& mesh, const float radius,
			const glm::ivec3& resolution, const bool inverted, const SampleMode sampleMode);
	}; 
}

#endif // !VOLUME_SAMPLER_H_