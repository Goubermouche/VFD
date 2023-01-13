#ifndef SEARCH_INFO_H
#define SEARCH_INFO_H

#include "pch.h"

namespace vfd
{
	struct SearchInfo
	{
		glm::vec3 GridMin;
		glm::vec3 GridDelta;

		glm::uvec3 GridDimension;
		glm::uvec3 MetaGridDimension;

		unsigned int ParticleCount;
		float SquaredSearchRadius;
	};
}

#endif // !SEARCH_INFO_H