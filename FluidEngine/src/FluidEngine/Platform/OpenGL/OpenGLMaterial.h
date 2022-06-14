#ifndef OPENGL_MATERIAL_H_
#define OPENGL_MATERIAL_H_

#include "FluidEngine/Renderer/Material.h"
#include "FluidEngine/Core/Buffer.h"

namespace fe::opengl {
	class OpenGLMaterial : public Material
	{
	public:
		OpenGLMaterial(Ref<Shader> shader, const std::string& name);
		virtual ~OpenGLMaterial() override;

		virtual void Set(const std::string& name, bool value) override;
		virtual void Set(const std::string& name, int value) override;
		virtual void Set(const std::string& name, uint64_t value) override;
		virtual void Set(const std::string& name, float value) override;
		virtual void Set(const std::string& name, const glm::vec2& value) override;
		virtual void Set(const std::string& name, const glm::vec3& value) override;
		virtual void Set(const std::string& name, const glm::vec4& value) override;
		virtual void Set(const std::string& name, const glm::mat3& value) override;
		virtual void Set(const std::string& name, const glm::mat4& value) override;

		virtual bool& GetBool(const std::string& name) override;
		virtual int32_t& GetInt(const std::string& name) override;
		virtual float& GetFloat(const std::string& name) override;
		virtual glm::vec2& GetVector2(const std::string& name) override;
		virtual glm::vec3& GetVector3(const std::string& name) override;
		virtual glm::vec4& GetVector4(const std::string& name) override;
		virtual glm::mat3& GetMatrix3(const std::string& name) override;
		virtual glm::mat4& GetMatrix4(const std::string& name) override;

		template <typename T>
		void Set(const std::string& name, const T& value)
		{
			auto decl = FindUniformDeclaration(name);
			ASSERT(decl, "could not find uniform '" + name + "'!");
			if (!decl) {
				return;
			}

			auto& buffer = m_UniformStorageBuffer;
			buffer.Write((byte*)&value, decl->GetSize(), decl->GetOffset());
		}

		template<typename T>
		T& Get(const std::string& name)
		{
			auto decl = FindUniformDeclaration(name);
			ASSERT(decl, "could not find uniform!");
			auto& buffer = m_UniformStorageBuffer;
			return buffer.Read<T>(decl->GetOffset());
		}

		virtual void Bind() override;
		virtual void Unbind() override;

		virtual Ref<Shader> GetShader() override;
		virtual const std::string& GetName() const override;
	private:
		/// <summary>
		/// Retrieves a uniform declaration from the specified name, if
		/// no uniform with that name exists an assert is triggered.
		/// </summary>
		/// <param name="name"></param>
		/// <returns></returns>
		const ShaderUniform* FindUniformDeclaration(const std::string& name);
	private:
		Ref<Shader> m_Shader;
		std::string m_Name;

		Buffer m_UniformStorageBuffer;
	};
}

#endif // !OPENGL_MATERIAL_H_