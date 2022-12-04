#ifndef DFSPH_PARTICLE_H
#define DFSPH_PARTICLE_H

#include "pch.h"

namespace vfd
{
	struct DFSPHParticle
	{
		glm::vec3 Position;
		glm::vec3 Velocity;
	};
}

#endif // !DFSPH_PARTICLE_H