#include "pch.h"
#include "Shader.h"

#include "FluidEngine/Renderer/RendererAPI.h"
#include "FluidEngine/Platform/OpenGL/OpenGLShader.h"

namespace fe {
	ShaderUniform::ShaderUniform(std::string name, ShaderDataType type, uint32_t size, uint32_t offset)
		: m_Name(std::move(name)), m_Type(type), m_Size(size), m_Offset(offset)
	{
	}

	Ref<Shader> Shader::Create(const std::string& filePath)
	{
		switch (RendererAPI::GetAPIType())
		{
		case RendererAPIType::None:    return nullptr;
		case RendererAPIType::OpenGL:  return Ref<opengl::OpenGLShader>::Create(filePath);
		}

		ASSERT(false, "unsupported rendering API!");
		return nullptr;
	}
}