#ifndef SPIRV_SHADER_H_
#define SPIRV_SHADER_H_

namespace fe {
	class SPIRVShader : public RefCounted
	{
	public:
		SPIRVShader(const std::string& filepath);
		~SPIRVShader();

		void Bind() const;
		void Unbind() const;

		void SetInt(const std::string& name, int value);
		void SetIntArray(const std::string& name, int* values, uint32_t count);
		void SetFloat(const std::string& name, float value);
		void SetFloat2(const std::string& name, const glm::vec2& value);
		void SetFloat3(const std::string& name, const glm::vec3& value);
		void SetFloat4(const std::string& name, const glm::vec4& value);
		void SetMat4(const std::string& name, const glm::mat4& value);

		void UploadUniformInt(const std::string& name, int value);
		void UploadUniformIntArray(const std::string& name, int* values, uint32_t count);
		void UploadUniformFloat(const std::string& name, float value);
		void UploadUniformFloat2(const std::string& name, const glm::vec2& value);
		void UploadUniformFloat3(const std::string& name, const glm::vec3& value);
		void UploadUniformFloat4(const std::string& name, const glm::vec4& value);
		void UploadUniformMat3(const std::string& name, const glm::mat3& matrix);
		void UploadUniformMat4(const std::string& name, const glm::mat4& matrix);
	private:
		std::string ReadFile(const std::string& filepath);
		std::unordered_map<unsigned int, std::string> PreProcess(const std::string& source);

		void CompileOrGetVulkanBinaries(const std::unordered_map<unsigned int, std::string>& shaderSources);
		void Reflect(unsigned int stage, const std::vector<uint32_t>& shaderData);
		void CompileOrGetOpenGLBinaries();
		void CreateProgram();
	private:
		uint32_t m_RendererID;
		std::string m_FilePath;
		std::string m_Name;

		std::unordered_map<unsigned int, std::vector<uint32_t>> m_VulkanSPIRV;
		std::unordered_map<unsigned int, std::vector<uint32_t>> m_OpenGLSPIRV;

		std::unordered_map<unsigned int, std::string> m_OpenGLSourceCode;
	};
}

#endif // !SPIRV_SHADER_H_