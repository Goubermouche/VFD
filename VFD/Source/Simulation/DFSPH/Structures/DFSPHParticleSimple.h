#ifndef DFSPH_PARTICLE_SIMPLE_H
#define DFSPH_PARTICLE_SIMPLE_H

#include "pch.h"

namespace vfd
{
	struct DFSPHParticleSimple
	{
		glm::vec3 Position;
		glm::vec3 Velocity;
		glm::vec3 Acceleration;
	};
}

#endif // !DFSPH_PARTICLE_SIMPLE_H