#ifndef GRID_INFO_H
#define GRID_INFO_H

#include "pch.h"

namespace vfdcu {
	struct GridInfo
	{
		glm::vec3 GridMin;
		glm::vec3 GridDelta;
		glm::uvec3 GridDimension;
		glm::uvec3 MetaGridDimension;

		unsigned int ParticleCount;
		float SquaredSearchRadius;
	};
}

#endif // !GRID_INFO_H