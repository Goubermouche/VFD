#ifndef BB_H
#define BB_H

namespace fe {
	/// <summary>
	/// A non-axis aligned bounding box, defined by it's minima and maxima. 
	/// </summary>
	class BoundingBox
	{
	public:
		BoundingBox() = default;
		BoundingBox(const std::vector<glm::vec3>& vertices);
		~BoundingBox() = default;

		void SetEmpty();
		void Extend(const glm::vec3& vec);

		glm::dvec3 Diagonal();
		[[nodiscard]]
		glm::dvec3 Diagonal() const;

		[[nodiscard]]
		bool Contains(const glm::dvec3& vec) const;
	public:
		glm::dvec3 min = {  DBL_MAX,  DBL_MAX,  DBL_MAX };
		glm::dvec3 max = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
	};
}

#endif // !BB_H