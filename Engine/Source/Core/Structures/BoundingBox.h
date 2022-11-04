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
		BoundingBox(const std::vector<glm::dvec3>& vertices);
		BoundingBox(glm::dvec3 min, glm::dvec3 max);
		~BoundingBox() = default;

		void SetEmpty();
		void Extend(const glm::dvec3& vec);

		glm::dvec3 Diagonal();
		glm::dvec3 Diagonal() const;

		bool Contains(const glm::dvec3& vec) const;
	public:
		glm::dvec3 min = {  DBL_MAX,  DBL_MAX,  DBL_MAX };
		glm::dvec3 max = { -DBL_MAX, -DBL_MAX, -DBL_MAX };
	};
}

#endif // !BB_H