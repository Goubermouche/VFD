#ifndef SPIRV_SHADER_H
#define SPIRV_SHADER_H

#include "Renderer/Buffers/UniformBuffer.h"
#include <Glad/glad.h>

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
		ShaderUniform(std::string name, ShaderDataType type, unsigned int size, unsigned int offset);
		~ShaderUniform() = default;

		[[nodiscard]]
		const std::string& GetName() const {
			return m_Name;
		}

		[[nodiscard]]
		ShaderDataType GetType() const {
			return m_Type;
		}

		[[nodiscard]]
		unsigned int GetSize() const {
			return m_Size;
		}

		[[nodiscard]]
		unsigned int GetOffset() const {
			return m_Offset;
		}
	private:
		std::string m_Name;
		ShaderDataType m_Type = ShaderDataType::None;
		unsigned int m_Size = 0;
		unsigned int m_Offset = 0;
	};

	/// <summary>
	/// Representation of a shader uniform buffer. 
	/// </summary>
	struct ShaderBuffer {
		std::string Name;
		unsigned int Size = 0;
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
		static void Unbind();

		/// <summary>
		/// Gets a representation of the shader uniform buffers.
		/// </summary>
		/// <returns>Vector containing shader buffers.</returns>
		std::vector<ShaderBuffer>& GetShaderBuffers() {
			return m_Buffers;
		}

		std::string GetSourceFilepath() {
			return m_FilePath;
		}

		uint32_t GetRendererID() const
		{
			return m_RendererID;
		}
	private:
		std::string ReadFile(const std::string& filepath) const;
		std::unordered_map<GLenum, std::string> PreProcess(const std::string& source) const;

		void CompileOrGetVulkanBinaries(const std::unordered_map<GLenum, std::string>& shaderSources);
		void CompileOrGetOpenGLBinaries();

		/// <summary>
		/// Extracts the needed buffers from the shader after compilation. Currently, only the uniform buffers are extracted into two categories: regular and property buffers (property buffers are denoted by a 'Property' name).
		/// </summary>
		/// <param name="stage">Current shader stage.</param>
		/// <param name="shaderData">Shader byte code.</param>
		void Reflect(GLenum stage, const std::vector<uint32_t>& shaderData);
		void CreateProgram();
	private:
		uint32_t m_RendererID = 0;
		std::string m_FilePath; // Source file path

		std::unordered_map<GLenum, std::string> m_OpenGLSourceCode;
		std::vector<ShaderBuffer> m_Buffers;

		// Shader binaries
		std::unordered_map<GLenum, std::vector<uint32_t>> m_VulkanSPIRV;
		std::unordered_map<GLenum, std::vector<uint32_t>> m_OpenGLSPIRV;
	};

	// TODO: convert shaders into assets
	// TODO: implement internal and public shaders
	class ShaderLibrary {
	public:
		static Ref<Shader> GetShader(const std::string& filepath);
		static void AddShader(const std::string& filepath);

		static const std::unordered_map<std::string, Ref<Shader>>& GetShaders() {
			return m_Shaders;
		}

	private:
		static std::unordered_map<std::string, Ref<Shader>> m_Shaders;
	};
}

#endif // !SPIRV_SHADER_H