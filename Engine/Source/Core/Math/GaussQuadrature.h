#ifndef GAUSS_QUADRATURE
#define GAUSS_QUADRATURE

#include "Core/Structures/BoundingBox.h"

namespace fe {
	class GaussQuadrature {
	public:
		using Integrand = std::function<double(const glm::vec3&)>;
		using Domain = BoundingBox;

		static double Integrate(Integrand integrand, const Domain& domain, unsigned int p);
	};
}

#endif // !GAUSS_QUADRATURE
