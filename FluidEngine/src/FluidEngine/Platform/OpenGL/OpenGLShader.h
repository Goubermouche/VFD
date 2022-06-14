#ifndef OPENGL_SHADER_H_
#define OPENGL_SHADER_H_

#include "FluidEngine/Renderer/Shader.h"

namespace fe::opengl {
	class OpenGLShader : public Shader
	{
	public:
		OpenGLShader(const std::string& filePath);
		virtual ~OpenGLShader() override;

		virtual void Bind() override;
		virtual void Unbind() override;

		virtual const std::unordered_map<std::string, ShaderBuffer>& GetShaderBuffers() const override { return m_Buffers; }
		virtual const uint32_t GetUniformBuffer() const override { return m_UniformBuffer; }

	private:
		ShaderProgramSource Parse(const std::string& filePath) const;
		uint32_t Compile(const uint32_t type, const std::string& source) const;
		uint32_t CreateProgram(const std::string& vertexShader, const std::string& fragmentShader, const std::string& geometryShader) const;

		ShaderDataType GetShaderDataTypeFromGLenum(uint32_t type) {
			switch (type)
			{
			case 0x8B56: return ShaderDataType::Bool;
			case 0x1404: return ShaderDataType::Int;
			case 0x1405: return ShaderDataType::Uint;
			case 0x1406: return ShaderDataType::Float;
			case 0x8B50: return ShaderDataType::Float2;
			case 0x8B51: return ShaderDataType::Float3;
			case 0x8B52: return ShaderDataType::Float4;
			case 0x8B5B: return ShaderDataType::Mat3;
			case 0x8B5C: return ShaderDataType::Mat4;
			}

			ERROR(type);
			ERROR("unknown shader data type!");

			return ShaderDataType::None;
		}
		uint32_t GetShaderDataTypeSize(ShaderDataType type) const {
			switch (type)
			{
			case ShaderDataType::Bool:   return 4;
			case ShaderDataType::Int:    return 4;
			case ShaderDataType::Uint:   return 4;
			case ShaderDataType::Float:  return 4;
			case ShaderDataType::Float2: return 8;
			case ShaderDataType::Float3: return 12;
			case ShaderDataType::Float4: return 16;
			case ShaderDataType::Mat3:   return 3 * 3 * 4;
			case ShaderDataType::Mat4:   return 4 * 4 * 4;
			}

			ERROR("unknown shader data type!");
			return 0;
		}
	private:
		uint32_t m_RendererID;
		uint32_t m_UniformBuffer;

		std::unordered_map<std::string, ShaderBuffer> m_Buffers;

		uint32_t m_BlockIndex;
		std::string mFilePath;

		enum OpenGLShaderType {
			None = -1,
			Vertex = 0,
			Fragment = 1,
			Geometry = 2
		};
	};
}

#endif // !OPENGL_SHADER_H_