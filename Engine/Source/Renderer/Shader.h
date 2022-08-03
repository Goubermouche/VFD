#ifndef SPIRV_SHADER_H_
#define SPIRV_SHADER_H_

#include "Renderer/Buffers/UniformBuffer.h"

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

	/// <summary>
	/// Representation of a shader uniform buffer variable. 
	/// </summary>
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

	/// <summary>
	/// Representation of a shader uniform buffer. 
	/// </summary>
	struct ShaderBuffer {
		std::string Name;
		uint32_t Size = 0;
		bool IsPropertyBuffer = false;
		Ref<UniformBuffer> Buffer;
		std::unordered_map<std::string, ShaderUniform> Uniforms;
	};

	class Shader : public RefCounted
	{
	public:
		Shader(const std::string& filepath);
		~Shader();

		void Bind() const;
		void Unbind() const;

		/// <summary>
		/// Gets a representation of the shader uniform buffers.
		/// </summary>
		/// <returns>Vector containing shader buffers.</returns>
		std::vector<ShaderBuffer>& GetShaderBuffers() {
			return m_Buffers;
		}

		inline std::string GetSourceFilepath() {
			return m_FilePath;
		}
	private:
		std::string ReadFile(const std::string& filepath);
		std::unordered_map<uint32_t, std::string> PreProcess(const std::string& source);

		void CompileOrGetVulkanBinaries(const std::unordered_map<uint32_t, std::string>& shaderSources);
		void CompileOrGetOpenGLBinaries();

		/// <summary>
		/// Extracts the needed buffers from the shader after compilation. Currently, only the uniform buffers are extracted into two categories: regular and property buffers (property buffers are denoted by a 'Property' name).
		/// </summary>
		/// <param name="stage">Current shader stage.</param>
		/// <param name="shaderData">Shader byte code.</param>
		void Reflect(uint32_t stage, const std::vector<uint32_t>& shaderData);
		void CreateProgram();
	private:
		uint32_t m_RendererID;
		std::string m_FilePath; // Source file path

		// Shader binaries
		std::unordered_map<uint32_t, std::vector<uint32_t>> m_VulkanSPIRV;
		std::unordered_map<uint32_t, std::vector<uint32_t>> m_OpenGLSPIRV;

		std::unordered_map<uint32_t, std::string> m_OpenGLSourceCode;
		std::vector<ShaderBuffer> m_Buffers;
	};

	class ShaderLibrary {
	public:
		static Ref<Shader> GetShader(const std::string& filepath);
		static void AddShader(const std::string& filepath);
	private:
		static std::unordered_map<std::string, Ref<Shader>> m_Shaders;
	};
}

#endif // !SPIRV_SHADER_H_