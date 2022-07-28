#include "pch.h"
#include "Material.h"

#include <Glad/glad.h>

namespace fe {
	Material::Material(Ref<Shader> shader, const std::string& name)
		: m_Shader(shader), m_Name(name)
	{
		const auto& shaderBuffers = m_Shader->GetShaderBuffers();

		for (auto const& [key, shaderBuffer] : shaderBuffers) {

			auto& buffer = m_UniformStorageBuffers[shaderBuffer.Name];
			buffer.StorageBuffer.Allocate(shaderBuffer.Size);
			buffer.StorageBuffer.ZeroInitialize();
		}
	}

	Material::~Material()
	{
		for (auto& [key, buffer] : m_UniformStorageBuffers) {
			buffer.StorageBuffer.Release();
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
		uint32_t uniformBufferRendererID = m_Shader->GetUniformBuffers().at("Data");
		m_Shader->Bind();

		for (auto [key, buffer] : m_Shader->GetUniformBuffers()) {
			auto& storage = m_UniformStorageBuffers.at(key);

			if (storage.ValueChanged) {
				buffer->SetData(storage.StorageBuffer.Data, storage.StorageBuffer.Size, 0);
				storage.ValueChanged = false;
			}
		}

		// TODO: keep track of updated buffers
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

	const std::pair<UniformStorageBuffer*, const ShaderUniform*> Material::FindUniformDeclaration(const std::string& name)
	{
		const auto& shaderBuffers = m_Shader->GetShaderBuffers();

		for (auto& [key, buffer] : shaderBuffers) {
			if (buffer.uniforms.contains(name)) {
				return { &m_UniformStorageBuffers[buffer.Name], &buffer.uniforms.at(name)};
			}
		}

		return { nullptr, nullptr };
	}
}