#ifndef BB_H
#define BB_H

#include "pch.h"

namespace vfd {
	/// <summary>
	/// A non-axis aligned bounding box, defined by it's extremes. Note that 
	/// only glm vectors are currently supported.
	/// </summary>
	template<typename T = glm::vec3>
	class BoundingBox
	{
	public:
		__host__ __device__ BoundingBox() = default;
		__host__ __device__ BoundingBox(const std::vector<T>& vertices) {
			for (const T& vertex : vertices)
			{
				Extend(vertex);
			}
		}

		__host__ __device__ BoundingBox(const T& min, const T& max)
			: min(min), max(max) 
		{}

		__host__ __device__ ~BoundingBox() = default;

		__host__ __device__ void Extend(const T& vec) {
			min = glm::min(min, vec);
			max = glm::max(max, vec);
		}

		__host__ __device__ T Diagonal() {
			return max - min;
		}

		__host__ __device__ T Diagonal() const {
			return max - min;
		}

		__host__ __device__ bool Contains(const T& vec) const {
			return min.x <= vec.x && min.y <= vec.y && min.z <= vec.z && max.x >= vec.x && max.y >= vec.y && max.z >= vec.z;
		}
	public:
		T min = T();
		T max = T();
	};
}

#endif // !BB_H