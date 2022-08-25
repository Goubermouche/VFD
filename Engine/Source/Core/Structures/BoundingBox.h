#ifndef BB_H
#define BB_H

namespace fe {
	class BoundingBox
	{
	public:
		BoundingBox() = default;
		BoundingBox(const std::vector<glm::vec3>& vertices);
		~BoundingBox() = default;

		void SetEmpty();
		void Extend(const glm::vec3& vec);

		glm::vec3 Diagonal();
		[[nodiscard]]
		glm::vec3 Diagonal() const;

		[[nodiscard]]
		bool Contains(const glm::vec3& vec) const;
	public:
		glm::vec3 min = { FLT_MAX, FLT_MAX, FLT_MAX };
		glm::vec3 max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	};
}

#endif // !BB_H