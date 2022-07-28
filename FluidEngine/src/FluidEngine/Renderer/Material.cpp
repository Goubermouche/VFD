#include "pch.h"
#include "Material.h"

#include <Glad/glad.h>

namespace fe {
	Material::Material(Ref<Shader> shader)
		: m_Shader(shader)
	{
		const auto& shaderBuffers = m_Shader->GetShaderBuffers();

		for (size_t i = 0; i < shaderBuffers.size(); i++)
		{
			auto& localBuffer = m_Buffers.emplace_back();
			//localBuffer.Value.Allocate();
			//localBuffer.Value.Fill(1); // TODO: check if 0 is better
			localBuffer.IsPropertyBuffer = shaderBuffers[i].IsPropertyBuffer;

			localBuffer.Value.resize(shaderBuffers[i].Size, {});

		}
	}

	Material::~Material()
	{}

	void Material::SetPropertyBuffer(std::vector<std::byte>& buffer)
	{
		for (size_t i = 0; i < m_Buffers.size(); i++)
		{
			if (m_Buffers[i].IsPropertyBuffer) {
				m_Buffers[i].Value = buffer;
				m_Buffers[i].ValueChanged = true;
				return;
			}
		}
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

		auto& shaderBuffers = m_Shader->GetShaderBuffers();

		for (size_t i = 0; i < shaderBuffers.size(); i++)
		{
			auto& localBuffer = m_Buffers[i];

			if (localBuffer.ValueChanged) {
				shaderBuffers[i].Buffer->SetData(localBuffer.Value.data(), localBuffer.Value.size(), 0);
				localBuffer.ValueChanged = false;
			}
		}
	}

	void Material::Unbind()
	{
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}

	const std::pair<MaterialBuffer*, const ShaderUniform*> Material::FindUniformDeclaration(const std::string& name)
	{
		const auto& shaderBuffers = m_Shader->GetShaderBuffers();

		for (size_t i = 0; i < shaderBuffers.size(); i++)
		{
			if (shaderBuffers[i].Uniforms.contains(name)) {
				return { &m_Buffers[i], &shaderBuffers[i].Uniforms.at(name) };
			}
		}

		return { nullptr, nullptr };
	}
}