#ifndef GAUSS_QUADRATURE
#define GAUSS_QUADRATURE

#include "Core/Structures/BoundingBox.h"

namespace fe {
	class GaussQuadrature {
	public:
		using Integrand = std::function<double(const glm::vec3&)>;
		
		static double Integrate(const Integrand& integrand, const BoundingBox& domain, unsigned int p);
	};
}

#endif // !GAUSS_QUADRATURE
