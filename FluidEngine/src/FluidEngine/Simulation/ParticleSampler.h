
#ifndef PARTICLE_SAMPLER_H_
#define PARTICLE_SAMPLER_H_

namespace fe {
	class ParticleSampler
	{
		static void SampleVolume(const unsigned int vertexCount,
			const glm::vec3* vertices, const unsigned int faceCount, const unsigned int* faces,
			const size_t particleRadius, const std::array<unsigned int, 3>& resolution,
			const bool inverted, const unsigned int sampleMode,
			std::vector<glm::vec3>& samples);
	};
}

#endif // !PARTICLE_SAMPLER_H_