#include "pch.h"
#include "Shader.h"

#include <Glad/glad.h>

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_glsl.hpp>

namespace fe {
	static ShaderDataType SPIRTypeToShaderDataType(spirv_cross::SPIRType type) {
		switch (type.basetype)
		{
		case spirv_cross::SPIRType::Boolean:  return ShaderDataType::Bool;
		case spirv_cross::SPIRType::Int:
			if (type.vecsize == 1)            return ShaderDataType::Int;
		case spirv_cross::SPIRType::UInt:     return ShaderDataType::Uint;
		case spirv_cross::SPIRType::Float:
			if (type.columns == 3)            return ShaderDataType::Mat3;
			if (type.columns == 4)            return ShaderDataType::Mat4;
			if (type.vecsize == 1)            return ShaderDataType::Float;
			if (type.vecsize == 2)            return ShaderDataType::Float2;
			if (type.vecsize == 3)            return ShaderDataType::Float3;
			if (type.vecsize == 4)            return ShaderDataType::Float4;
			break;
		}

		ASSERT("unknown type!");
		return ShaderDataType::None;
	}

	static const char* GLShaderStageToString(GLenum stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:   return "GL_VERTEX_SHADER";
		case GL_FRAGMENT_SHADER: return "GL_FRAGMENT_SHADER";
		}
		ASSERT("unknown shader stage!");
		return nullptr;
	}

	static shaderc_shader_kind GLShaderStageToShaderC(GLenum stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:   return shaderc_glsl_vertex_shader;
		case GL_FRAGMENT_SHADER: return shaderc_glsl_fragment_shader;
		}
		ASSERT("unknown shader stage!");
		return (shaderc_shader_kind)0;
	}

	static const char* GLShaderStageCachedOpenGLFileExtension(uint32_t stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:    return ".CachedOpenGL.vert";
		case GL_FRAGMENT_SHADER:  return ".CachedOpenGL.frag";
		}
		ASSERT("unknown shader stage!");
		return "";
	}

	static const char* GLShaderStageCachedVulkanFileExtension(uint32_t stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:    return ".CachedVulkan.vert";
		case GL_FRAGMENT_SHADER:  return ".CachedVulkan.frag";
		}

		ASSERT("unknown shader stage!");
		return "";
	}

	static GLenum ShaderTypeFromString(const std::string& type)
	{
		if (type == "vertex") {
			return GL_VERTEX_SHADER;
		}
		if (type == "fragment" || type == "pixel") {
			return GL_FRAGMENT_SHADER;
		}

		ASSERT(false, "unknown shader type!");
		return 0;
	}

	static const char* GetCacheDirectory()
	{
		return "res/Shaders/Cache";
	}

	static void CreateCacheDirectoryIfNeeded()
	{
		std::string cacheDirectory = GetCacheDirectory();
		if (!std::filesystem::exists(cacheDirectory)) {
			std::filesystem::create_directories(cacheDirectory);
		}
	}

	Shader::Shader(const std::string& filepath)
		: m_FilePath(filepath)
	{
		std::string source = ReadFile(filepath);
		auto shaderSources = PreProcess(source);

		CompileOrGetVulkanBinaries(shaderSources);
		CompileOrGetOpenGLBinaries();
		CreateProgram();

		LOG("shader '" + m_FilePath + "' created successfully", "renderer][shader", ConsoleColor::Green);
	}

	Shader::~Shader()
	{
		glDeleteProgram(m_RendererID);
	}

	void Shader::Bind() const
	{
		glUseProgram(m_RendererID);
	}

	void Shader::Unbind() const
	{
		glUseProgram(0);
	}

	std::string Shader::ReadFile(const std::string& filepath)
	{
		std::string result;
		std::ifstream in(filepath, std::ios::in | std::ios::binary); // ifstream closes itself due to RAII
		if (in)
		{
			in.seekg(0, std::ios::end);
			size_t size = in.tellg();
			if (size != -1)
			{
				result.resize(size);
				in.seekg(0, std::ios::beg);
				in.read(&result[0], size);
			}
			else
			{
				ERR("could not read file '" + filepath + "'");
			}
		}
		else
		{
			ERR("could not open file '" + filepath + "'");
		}

		return result;
	}

	std::unordered_map<unsigned int, std::string> Shader::PreProcess(const std::string& source)
	{
		std::unordered_map<unsigned int, std::string> shaderSources;

		const char* typeToken = "#type";
		size_t typeTokenLength = strlen(typeToken);
		size_t pos = source.find(typeToken, 0); //Start of shader type declaration line

		while (pos != std::string::npos)
		{
			size_t eol = source.find_first_of("\r\n", pos); //End of shader type declaration line
			ASSERT(eol != std::string::npos, "syntax error");
			size_t begin = pos + typeTokenLength + 1; //Start of shader type name (after "#type " keyword)
			std::string type = source.substr(begin, eol - begin);
			ASSERT(ShaderTypeFromString(type), "invalid shader type specified");

			size_t nextLinePos = source.find_first_not_of("\r\n", eol); //Start of shader code after shader type declaration line
			ASSERT(nextLinePos != std::string::npos, "syntax error");
			pos = source.find(typeToken, nextLinePos); //Start of next shader type declaration line

			shaderSources[ShaderTypeFromString(type)] = (pos == std::string::npos) ? source.substr(nextLinePos) : source.substr(nextLinePos, pos - nextLinePos);
		}

		return shaderSources;
	}

	void Shader::CompileOrGetVulkanBinaries(const std::unordered_map<unsigned int, std::string>& shaderSources)
	{
		unsigned int program = glCreateProgram();

		shaderc::Compiler compiler;
		shaderc::CompileOptions options;
		options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
		const bool optimize = true;

		if (optimize) {
			options.SetOptimizationLevel(shaderc_optimization_level_performance);
			options.SetGenerateDebugInfo();
		}

		std::filesystem::path cacheDirectory = GetCacheDirectory();
		auto& shaderData = m_VulkanSPIRV;
		shaderData.clear();

		for (auto&& [stage, source] : shaderSources)
		{
			std::filesystem::path shaderFilePath = m_FilePath;
			std::filesystem::path cachedPath = cacheDirectory / (shaderFilePath.filename().string() + GLShaderStageCachedVulkanFileExtension(stage));

			std::ifstream in(cachedPath, std::ios::in | std::ios::binary);

			// Check shader chache
			if (in.is_open())
			{
				in.seekg(0, std::ios::end);
				auto size = in.tellg();
				in.seekg(0, std::ios::beg);

				auto& data = shaderData[stage];
				data.resize(size / sizeof(uint32_t));
				in.read((char*)data.data(), size);
			}
			// Shader is not cached, cache it
			else
			{
				shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(source, GLShaderStageToShaderC(stage), m_FilePath.c_str(), options);
				if (module.GetCompilationStatus() != shaderc_compilation_status_success)
				{
					ASSERT(module.GetErrorMessage());
				}

				shaderData[stage] = std::vector<uint32_t>(module.cbegin(), module.cend());

				std::ofstream out(cachedPath, std::ios::out | std::ios::binary);
				if (out.is_open())
				{
					auto& data = shaderData[stage];
					out.write((char*)data.data(), data.size() * sizeof(uint32_t));
					out.flush();
					out.close();
				}
			}
		}

		for (auto&& [stage, data] : shaderData) {
			Reflect(stage, data);
		}
	}

	void Shader::CompileOrGetOpenGLBinaries()
	{
		auto& shaderData = m_OpenGLSPIRV;
		shaderc::Compiler compiler;
		shaderc::CompileOptions options;
		options.SetTargetEnvironment(shaderc_target_env_opengl, shaderc_env_version_opengl_4_5);
		const bool optimize = false;

		if (optimize) {
			options.SetOptimizationLevel(shaderc_optimization_level_performance);
		}

		std::filesystem::path cacheDirectory = GetCacheDirectory();

		shaderData.clear();
		m_OpenGLSourceCode.clear();

		for (auto&& [stage, spirv] : m_VulkanSPIRV)
		{
			std::filesystem::path shaderFilePath = m_FilePath;
			std::filesystem::path cachedPath = cacheDirectory / (shaderFilePath.filename().string() + GLShaderStageCachedOpenGLFileExtension(stage));

			std::ifstream in(cachedPath, std::ios::in | std::ios::binary);
			if (in.is_open())
			{
				in.seekg(0, std::ios::end);
				auto size = in.tellg();
				in.seekg(0, std::ios::beg);

				auto& data = shaderData[stage];
				data.resize(size / sizeof(uint32_t));
				in.read((char*)data.data(), size);
			}
			else
			{
				spirv_cross::CompilerGLSL glslCompiler(spirv);
				m_OpenGLSourceCode[stage] = glslCompiler.compile();
				auto& source = m_OpenGLSourceCode[stage];

				shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(source, GLShaderStageToShaderC(stage), m_FilePath.c_str());
				if (module.GetCompilationStatus() != shaderc_compilation_status_success)
				{
					ASSERT(module.GetErrorMessage());
				}

				shaderData[stage] = std::vector<uint32_t>(module.cbegin(), module.cend());

				std::ofstream out(cachedPath, std::ios::out | std::ios::binary);
				if (out.is_open())
				{
					auto& data = shaderData[stage];
					out.write((char*)data.data(), data.size() * sizeof(uint32_t));
					out.flush();
					out.close();
				}
			}
		}
	}

	void Shader::CreateProgram()
	{
		unsigned int program = glCreateProgram();

		std::vector<GLuint> shaderIDs;
		for (auto&& [stage, spirv] : m_OpenGLSPIRV)
		{
			GLuint shaderID = shaderIDs.emplace_back(glCreateShader(stage));
			glShaderBinary(1, &shaderID, GL_SHADER_BINARY_FORMAT_SPIR_V, spirv.data(), spirv.size() * sizeof(uint32_t));
			glSpecializeShader(shaderID, "main", 0, nullptr, nullptr);
			glAttachShader(program, shaderID);
		}

		glLinkProgram(program);
		int isLinked;
		glGetProgramiv(program, GL_LINK_STATUS, &isLinked);

		if (isLinked == GL_FALSE)
		{
			GLint maxLength;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

			std::vector<GLchar> infoLog(maxLength);
			glGetProgramInfoLog(program, maxLength, &maxLength, infoLog.data());
			ERR("failed to link shader '" + m_FilePath + "'");
			ERR(infoLog.data());

			glDeleteProgram(program);

			for (auto id : shaderIDs) {
				glDeleteShader(id);
			}
		}

		for (auto id : shaderIDs)
		{
			glDetachShader(program, id);
			glDeleteShader(id);
		}

		m_RendererID = program;
	}

	ShaderUniform::ShaderUniform(std::string name, ShaderDataType type, uint32_t size, uint32_t offset)
		: m_Name(std::move(name)), m_Type(type), m_Size(size), m_Offset(offset)
	{}

	void Shader::Reflect(unsigned int stage, const std::vector<uint32_t>& shaderData)
	{
		spirv_cross::Compiler compiler(shaderData);
		spirv_cross::ShaderResources resources = compiler.get_shader_resources();
		uint32_t bindingIndex = 0;
		// Uniform buffers
		for (const auto& resource : resources.uniform_buffers) {
			auto& bufferType = compiler.get_type(resource.base_type_id);

			const std::string& bufferName = resource.name;
			uint32_t bufferSize = (uint32_t)compiler.get_declared_struct_size(bufferType);

			ShaderBuffer& buffer = m_Buffers.emplace_back();
			buffer.Name = bufferName;
			buffer.Size = bufferSize;
			buffer.IsPropertyBuffer = bufferName == "Properties";

			uint32_t memberCount = (uint32_t)bufferType.member_types.size();

			// Member data
			for (size_t i = 0; i < memberCount; i++)
			{
				ShaderDataType uniformType = SPIRTypeToShaderDataType(compiler.get_type(bufferType.member_types[i]));
				std::string uniformName = compiler.get_member_name(bufferType.self, i);
				uint32_t uniformSize = (uint32_t)compiler.get_declared_struct_member_size(bufferType, i);
				uint32_t uniformOffset = compiler.type_struct_member_offset(bufferType, i);
				buffer.Uniforms[uniformName] = ShaderUniform(uniformName, uniformType, uniformSize, uniformOffset);
			}

			buffer.Buffer = Ref<UniformBuffer>::Create(bufferSize, bindingIndex);
			bindingIndex++;
		}
	}
}