#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "FluidEngine/Renderer/Shader.h"

namespace fe {
	class Material : public RefCounted
	{
	public:
		virtual ~Material() {}

		virtual void Set(const std::string& name, bool value) = 0;
		virtual void Set(const std::string& name, int value) = 0;
		virtual void Set(const std::string& name, uint64_t value) = 0;
		virtual void Set(const std::string& name, float value) = 0;
		virtual void Set(const std::string& name, const glm::vec2& value) = 0;
		virtual void Set(const std::string& name, const glm::vec3& value) = 0;
		virtual void Set(const std::string& name, const glm::vec4& value) = 0;
		virtual void Set(const std::string& name, const glm::mat3& value) = 0;
		virtual void Set(const std::string& name, const glm::mat4& value) = 0;

		virtual bool& GetBool(const std::string& name) = 0;
		virtual int32_t& GetInt(const std::string& name) = 0;
		virtual float& GetFloat(const std::string& name) = 0;
		virtual glm::vec2& GetVector2(const std::string& name) = 0;
		virtual glm::vec3& GetVector3(const std::string& name) = 0;
		virtual glm::vec4& GetVector4(const std::string& name) = 0;
		virtual glm::mat3& GetMatrix3(const std::string& name) = 0;
		virtual glm::mat4& GetMatrix4(const std::string& name) = 0;

		virtual void Bind() = 0;
		virtual void Unbind() = 0;

		virtual Ref<Shader> GetShader() = 0;
		virtual const std::string& GetName() const = 0;

		static Ref<Material> Create(const Ref<Shader>& shader, const std::string& name = "material");
	};
}

#endif // !MATERIAL_H_