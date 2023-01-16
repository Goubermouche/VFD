#ifndef MATERIAL_H
#define MATERIAL_H

#include "Renderer/Shader.h"

// TODO: add support for multiple property buffers
namespace vfd {
	struct MaterialBuffer {
		std::vector<unsigned char> Value;
		// bool ValueChanged = false;
		bool IsPropertyBuffer = false;
	};
	/// <summary>
	/// Simple Material class, holds a buffer containing shader settings and a reference to the shader.
	/// </summary>
	class Material : public RefCounted
	{
	public:
		Material(Ref<Shader> shader);
		~Material() = default;

		/// <summary>
		/// Sets the property buffer, if one exists. 
		/// </summary>
		/// <param name="buffer">Buffer of bytes. </param>
		void SetPropertyBuffer(std::vector<unsigned char>& buffer);

		// Setters
		void Set(const std::string& name, bool value);
		void Set(const std::string& name, int value);
		void Set(const std::string& name, uint64_t value);
		void Set(const std::string& name, uint32_t value);
		void Set(const std::string& name, float value);
		void Set(const std::string& name, const glm::vec2& value);
		void Set(const std::string& name, const glm::vec3& value);
		void Set(const std::string& name, const glm::vec4& value);
		void Set(const std::string& name, const glm::mat3& value);
		void Set(const std::string& name, const glm::mat4& value);

		// Getters
		bool&      GetBool(const std::string& name);
		int32_t&   GetInt(const std::string& name);
		float&     GetFloat(const std::string& name);
		glm::vec2& GetVector2(const std::string& name);
		glm::vec3& GetVector3(const std::string& name);
		glm::vec4& GetVector4(const std::string& name);
		glm::mat3& GetMatrix3(const std::string& name);
		glm::mat4& GetMatrix4(const std::string& name);

		Ref<Shader> GetShader() const;

		/// <summary>
		/// Returns a representation of the shader uniform buffers. 
		/// </summary>
		/// <returns>Vector containing material buffers.</returns>
		const std::vector<MaterialBuffer>& GetMaterialBuffers() const;

		void Bind();
		static void Unbind();
	private:
		template <typename T>
		void Set(const std::string& name, const T& value);

		template<typename T>
		T& Get(const std::string& name);

		/// <summary>
		/// Retrieves a uniform declaration using the specified name, if no uniform with that name exists a nullptr is returned.
		/// </summary>
		/// <param name="name">Uniform name.</param>
		/// <returns>Pair containing the parent buffer and the uniform itself.</returns>
		std::pair<MaterialBuffer*, const ShaderUniform*> GetUniformDeclaration(const std::string& name);
	private:
		Ref<Shader> m_Shader;
		std::vector<MaterialBuffer> m_Buffers;
	};

	template<typename T>
	inline void Material::Set(const std::string& name, const T& value)
	{
		const auto declaration = GetUniformDeclaration(name);
		ASSERT(declaration.first, "unable to find uniform '" + name + "' (" + m_Shader->GetSourceFilepath() + ")!");
		std::memcpy(declaration.first->Value.data() + declaration.second->GetOffset(), (unsigned char*)&value, declaration.second->GetSize());
		// decl.first->ValueChanged = true;
	}

	template<typename T>
	inline T& Material::Get(const std::string& name)
	{
		const auto declaration = GetUniformDeclaration(name);
		ASSERT(declaration.first, "unable to find uniform (" + name + ")!");
		return *(T*)((unsigned char*)declaration.first->Value.data() + declaration.second->GetOffset());
	}
}

#endif // !MATERIAL_H