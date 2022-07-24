#include "pch.h"
#include "OpenGLMaterial.h"

#include <Glad/glad.h>

namespace fe::opengl {
	OpenGLMaterial::OpenGLMaterial(Ref<Shader> shader, const std::string& name)
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

	OpenGLMaterial::~OpenGLMaterial()
	{
		m_UniformStorageBuffer.Release();
	}

	void OpenGLMaterial::Set(const std::string& name, bool value)
	{
		Set<bool>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, int value)
	{
		Set<int>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, uint64_t value)
	{
		Set<uint64_t>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, float value)
	{
		Set<float>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, const glm::vec2& value)
	{
		Set<glm::vec2>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, const glm::vec3& value)
	{
		Set<glm::vec3>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, const glm::vec4& value)
	{
		Set<glm::vec4>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, const glm::mat3& value)
	{
		Set<glm::mat3>(name, value);
	}

	void OpenGLMaterial::Set(const std::string& name, const glm::mat4& value)
	{
		Set<glm::mat4>(name, value);
	}

	bool& OpenGLMaterial::GetBool(const std::string& name)
	{
		return Get<bool>(name);
	}

	int32_t& OpenGLMaterial::GetInt(const std::string& name)
	{
		return Get<int32_t>(name);
	}

	float& OpenGLMaterial::GetFloat(const std::string& name)
	{
		return Get<float>(name);
	}

	glm::vec2& OpenGLMaterial::GetVector2(const std::string& name)
	{
		return Get<glm::vec2>(name);
	}

	glm::vec3& OpenGLMaterial::GetVector3(const std::string& name)
	{
		return Get<glm::vec3>(name);
	}

	glm::vec4& OpenGLMaterial::GetVector4(const std::string& name)
	{
		return Get<glm::vec4>(name);
	}

	glm::mat3& OpenGLMaterial::GetMatrix3(const std::string& name)
	{
		return Get<glm::mat3>(name);
	}

	glm::mat4& OpenGLMaterial::GetMatrix4(const std::string& name)
	{
		return Get<glm::mat4>(name);
	}

	void OpenGLMaterial::Bind()
	{
		m_Shader->Bind();
		glBindBuffer(GL_UNIFORM_BUFFER, m_Shader->GetUniformBuffer());
		glBufferData(GL_UNIFORM_BUFFER, m_UniformStorageBuffer.GetSize(), m_UniformStorageBuffer.Data, GL_STATIC_DRAW);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, m_Shader->GetUniformBuffer());
	}

	void OpenGLMaterial::Unbind()
	{
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	Ref<Shader> OpenGLMaterial::GetShader() const
	{
		return m_Shader;
	}

	const std::string& OpenGLMaterial::GetName() const
	{
		return m_Name;
	}

	const ShaderUniform* OpenGLMaterial::FindUniformDeclaration(const std::string& name)
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