#ifndef GAUSS_QUADRATURE
#define GAUSS_QUADRATURE

#include "Core/Structures/BoundingBox.h"

namespace vfd {
	class GaussQuadrature {
	public:
		using Integrand = std::function<double(const glm::vec3&)>;
		
		static double Integrate(const Integrand& integrand, const BoundingBox<glm::dvec3>& domain, unsigned int p);
	};
}

#endif // !GAUSS_QUADRATURE
