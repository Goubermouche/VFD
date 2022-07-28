#ifndef MATERIAL_H_
#define MATERIAL_H_

#include "FluidEngine/Renderer/Shader.h"
#include "FluidEngine/Core/Buffer.h";;

namespace fe {
	struct UniformStorageBuffer {
		bool ValueChanged;
		Buffer StorageBuffer;
	};

	/// <summary>
	/// Simple Material class, holds a buffer containing shader settings and a reference to the shader.
	/// </summary>
	class Material : public RefCounted
	{
	public:
		Material(Ref<Shader> shader);
		~Material();

		void Set(const std::string& name, bool value);
		void Set(const std::string& name, int value);
		void Set(const std::string& name, uint64_t value);
		void Set(const std::string& name, float value);
		void Set(const std::string& name, const glm::vec2& value);
		void Set(const std::string& name, const glm::vec3& value);
		void Set(const std::string& name, const glm::vec4& value);
		void Set(const std::string& name, const glm::mat3& value);
		void Set(const std::string& name, const glm::mat4& value);

		bool& GetBool(const std::string& name);
		int32_t& GetInt(const std::string& name);
		float& GetFloat(const std::string& name);
		glm::vec2& GetVector2(const std::string& name);
		glm::vec3& GetVector3(const std::string& name);
		glm::vec4& GetVector4(const std::string& name);
		glm::mat3& GetMatrix3(const std::string& name);
		glm::mat4& GetMatrix4(const std::string& name);

		template <typename T>
		void Set(const std::string& name, const T& value)
		{
			auto decl = FindUniformDeclaration(name);
			ASSERT(decl.first, "could not find uniform '" + name + "'!");
			decl.first->StorageBuffer.Write((byte*)&value, decl.second->GetSize(), decl.second->GetOffset());
			decl.first->ValueChanged = true;
		}

		template<typename T>
		T& Get(const std::string& name)
		{
			auto decl = FindUniformDeclaration(name);
			ASSERT(decl.first, "could not find uniform '" + name + "'!");

			return decl.first->StorageBuffer.Read<T>(decl.second->GetOffset());
		}

		void Bind();
		void Unbind();

		Ref<Shader> GetShader() const;
	private:
		/// <summary>
		/// Retrieves a uniform declaration from the specified name, if no uniform with that name exists an assert is triggered.
		/// </summary>
		/// <param name="name">Uniform name</param>
		/// <returns>The requested shader uniform.</returns>
		const std::pair<UniformStorageBuffer*, const ShaderUniform*> FindUniformDeclaration(const std::string& name);
	private:
		Ref<Shader> m_Shader;

		std::vector< UniformStorageBuffer> m_UniformStorageBuffers;
	};
}

#endif // !MATERIAL_H_