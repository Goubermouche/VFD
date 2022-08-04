#ifndef BOX_H
#define BOX_H

namespace fe {
	class BoundingBox
	{
	public:
		BoundingBox();

		void SetEmpty();
		void Extend(const glm::vec3& vec);

		glm::vec3 Diagonal();
		[[nodiscard]]
		glm::vec3 Diagonal() const;

		[[nodiscard]]
		bool Contains(const glm::vec3& vec) const;

		static BoundingBox ComputeBoundingBox(const std::vector<glm::vec3>& vertices);
	public:
		glm::vec3 min;
		glm::vec3 max;
	};
}

#endif // !BOX_H