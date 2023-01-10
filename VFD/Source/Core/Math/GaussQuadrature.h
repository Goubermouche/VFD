#ifndef GAUSS_QUADRATURE
#define GAUSS_QUADRATURE

#include "Core/Structures/BoundingBox.h"

namespace vfd {
	class GaussQuadrature {
	public:
		using Integrand = std::function<float(const glm::vec3&)>;
		
		static float Integrate(const Integrand& integrand, const BoundingBox<glm::vec3>& domain, unsigned int p);
	};
}

#endif // !GAUSS_QUADRATURE
