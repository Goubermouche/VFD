#include "pch.h"
#include "Material.h"

#include <Glad/glad.h>

namespace fe {
	Material::Material(Ref<Shader> shader, const std::string& name)
		: m_Shader(shader), m_Name(name)
	{
		const auto& shaderBuffers = m_Shader->GetShaderBuffers();
		if (shaderBuffers.size() > 0)
		{
			uint32_t size = 0;
			for (auto [name, shaderBuffer] : shaderBuffers) {
				size += shaderBuffer.Size;
			}

			m_UniformStorageBuffer.Allocate(size);
			m_UniformStorageBuffer.ZeroInitialize();
		}
	}

	Material::~Material()
	{
		m_UniformStorageBuffer.Release();
	}

	void Material::Set(const std::string& name, bool value)
	{
		Set<bool>(name, value);
	}

	void Material::Set(const std::string& name, int value)
	{
		Set<int>(name, value);
	}

	void Material::Set(const std::string& name, uint64_t value)
	{
		Set<uint64_t>(name, value);
	}

	void Material::Set(const std::string& name, float value)
	{
		Set<float>(name, value);
	}

	void Material::Set(const std::string& name, const glm::vec2& value)
	{
		Set<glm::vec2>(name, value);
	}

	void Material::Set(const std::string& name, const glm::vec3& value)
	{
		Set<glm::vec3>(name, value);
	}

	void Material::Set(const std::string& name, const glm::vec4& value)
	{
		Set<glm::vec4>(name, value);
	}

	void Material::Set(const std::string& name, const glm::mat3& value)
	{
		Set<glm::mat3>(name, value);
	}

	void Material::Set(const std::string& name, const glm::mat4& value)
	{
		Set<glm::mat4>(name, value);
	}

	bool& Material::GetBool(const std::string& name)
	{
		return Get<bool>(name);
	}

	int32_t& Material::GetInt(const std::string& name)
	{
		return Get<int32_t>(name);
	}

	float& Material::GetFloat(const std::string& name)
	{
		return Get<float>(name);
	}

	glm::vec2& Material::GetVector2(const std::string& name)
	{
		return Get<glm::vec2>(name);
	}

	glm::vec3& Material::GetVector3(const std::string& name)
	{
		return Get<glm::vec3>(name);
	}

	glm::vec4& Material::GetVector4(const std::string& name)
	{
		return Get<glm::vec4>(name);
	}

	glm::mat3& Material::GetMatrix3(const std::string& name)
	{
		return Get<glm::mat3>(name);
	}

	glm::mat4& Material::GetMatrix4(const std::string& name)
	{
		return Get<glm::mat4>(name);
	}

	void Material::Bind()
	{
		m_Shader->Bind();
		glBindBuffer(GL_UNIFORM_BUFFER, m_Shader->GetUniformBuffer());
		glBufferData(GL_UNIFORM_BUFFER, m_UniformStorageBuffer.GetSize(), m_UniformStorageBuffer.Data, GL_STATIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_Shader->GetUniformBuffer());
	}

	void Material::Unbind()
	{
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	Ref<Shader> Material::GetShader() const
	{
		return m_Shader;
	}

	const std::string& Material::GetName() const
	{
		return m_Name;
	}

	const ShaderUniform* Material::FindUniformDeclaration(const std::string& name)
	{
		const auto& shaderBuffers = m_Shader->GetShaderBuffers();
		ASSERT(shaderBuffers.size() <= 1, "max number of buffers is 1!");

		if (shaderBuffers.size() > 0)
		{
			const ShaderBuffer& buffer = (*shaderBuffers.begin()).second;
			if (buffer.uniforms.find(name) == buffer.uniforms.end()) {
				return nullptr;
			}
			return &buffer.uniforms.at(name);
		}
		return nullptr;
	}
}