#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "FluidEngine/Renderer/Shader.h"

namespace fe {
	struct MaterialBuffer {
		std::vector<std::byte> Value;
		bool ValueChanged = false;
		bool IsPropertyBuffer = false;
	};
	/// <summary>
	/// Simple Material class, holds a buffer containing shader settings and a reference to the shader.
	/// </summary>
	class Material : public RefCounted
	{
	public:
		Material(Ref<Shader> shader);
		~Material();

		void SetPropertyBuffer(std::vector<std::byte>& buffer);

		// Setters
		void Set(const std::string& name, bool value);
		void Set(const std::string& name, int value);
		void Set(const std::string& name, uint64_t value);
		void Set(const std::string& name, float value);
		void Set(const std::string& name, const glm::vec2& value);
		void Set(const std::string& name, const glm::vec3& value);
		void Set(const std::string& name, const glm::vec4& value);
		void Set(const std::string& name, const glm::mat3& value);
		void Set(const std::string& name, const glm::mat4& value);

		// Getters
		bool& GetBool(const std::string& name);
		int32_t& GetInt(const std::string& name);
		float& GetFloat(const std::string& name);
		glm::vec2& GetVector2(const std::string& name);
		glm::vec3& GetVector3(const std::string& name);
		glm::vec4& GetVector4(const std::string& name);
		glm::mat3& GetMatrix3(const std::string& name);
		glm::mat4& GetMatrix4(const std::string& name);

		Ref<Shader> GetShader() const {
			return m_Shader;
		}

		const std::vector<MaterialBuffer>& GetMaterialBuffers() const {
			return m_Buffers;
		}

		void Bind();
		void Unbind();
	private:
		template <typename T>
		void Set(const std::string& name, const T& value)
		{
			auto decl = FindUniformDeclaration(name);
			ASSERT(decl.first, "could not find uniform '" + name + "'!");
			std::memcpy(decl.first->Value.data() + decl.second->GetOffset(), (std::byte*)&value, decl.second->GetSize());
			decl.first->ValueChanged = true;
		}

		template<typename T>
		T& Get(const std::string& name)
		{
			auto decl = FindUniformDeclaration(name);
			ASSERT(decl.first, "could not find uniform '" + name + "'!");
			return *(T*)((std::byte*)decl.first->Value.data() + decl.second->GetOffset());
		}

		/// <summary>
		/// Retrieves a uniform declaration using the specified name, if no uniform with that name exists a nullptr is returned.
		/// </summary>
		/// <param name="name">Uniform name.</param>
		/// <returns>Pair containing the parent buffer and the uniform itself.</returns>
		const std::pair<MaterialBuffer*, const ShaderUniform*> FindUniformDeclaration(const std::string& name);
	private:
		Ref<Shader> m_Shader;
		std::vector<MaterialBuffer> m_Buffers;
	};
}

#endif // !MATERIAL_H_