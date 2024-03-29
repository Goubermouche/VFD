#ifndef VOLUME_SAMPLER_H
#define VOLUME_SAMPLER_H

#include "Renderer/Mesh/EdgeMesh.h"

namespace vfd {
	enum class SampleMode {
		MinDensity = 0,
		MediumDensity = 1, 
		MaxDensity = 2
	};

	class ParticleSampler
	{
	public:
		static std::vector<glm::vec3> SampleMeshVolume(const Ref<EdgeMesh>& mesh, float radius,
			const glm::uvec3& resolution, bool inverted, SampleMode sampleMode);
	}; 
}

#endif // !VOLUME_SAMPLER_H