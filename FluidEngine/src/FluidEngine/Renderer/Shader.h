#ifndef SPIRV_SHADER_H_
#define SPIRV_SHADER_H_

#include "FluidEngine/Renderer/Buffers/UniformBuffer.h"

namespace fe {
	enum class ShaderDataType {
		None = 0,
		Bool,
		Int,
		Uint,
		Float,
		Float2,
		Float3,
		Float4,
		Mat3,
		Mat4
	};

	class ShaderUniform {
	public:
		ShaderUniform() = default;
		ShaderUniform(std::string name, ShaderDataType type, uint32_t size, uint32_t offset);

		const std::string& GetName() const {
			return m_Name;
		}

		ShaderDataType GetType() const {
			return m_Type;
		}

		uint32_t GetSize() const {
			return m_Size;
		}

		uint32_t GetOffset() const {
			return m_Offset;
		}
	private:
		std::string m_Name;
		ShaderDataType m_Type = ShaderDataType::None;
		uint32_t m_Size = 0;
		uint32_t m_Offset = 0;
	};

	struct ShaderBuffer {
		std::string Name;
		uint32_t Size = 0;
		Ref<UniformBuffer> Buffer;
		std::unordered_map<std::string, ShaderUniform> Uniforms;
		bool IsPropertyBuffer = false;
	};

	class Shader : public RefCounted
	{
	public:
		Shader(const std::string& filepath);
		~Shader();

		void Bind() const;
		void Unbind() const;

		std::vector<ShaderBuffer>& GetShaderBuffers() {
			return m_Buffers;
		}

		inline std::string GetSourceFilepath() {
			return m_FilePath;
		}
	private:
		std::string ReadFile(const std::string& filepath);
		std::unordered_map<unsigned int, std::string> PreProcess(const std::string& source);

		void CompileOrGetVulkanBinaries(const std::unordered_map<unsigned int, std::string>& shaderSources);
		void CompileOrGetOpenGLBinaries();

		void Reflect(unsigned int stage, const std::vector<uint32_t>& shaderData);
		void CreateProgram();
	private:
		uint32_t m_RendererID;
		std::string m_FilePath;
		std::string m_Name;

		std::unordered_map<unsigned int, std::vector<uint32_t>> m_VulkanSPIRV;
		std::unordered_map<unsigned int, std::vector<uint32_t>> m_OpenGLSPIRV;

		std::unordered_map<unsigned int, std::string> m_OpenGLSourceCode;

		std::vector<ShaderBuffer> m_Buffers;
	};
}

#endif // !SPIRV_SHADER_H_