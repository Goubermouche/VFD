#include "pch.h"
#include "Shader.h"

#include <shaderc/shaderc.hpp>
#include <spirv_cross/spirv_cross.hpp>
#include <spirv_cross/spirv_glsl.hpp>

namespace vfd {
	static ShaderDataType SPIRTypeToShaderDataType(const spirv_cross::SPIRType type) {
		switch (type.basetype)
		{
		case spirv_cross::SPIRType::Boolean:          return ShaderDataType::Bool;
		case spirv_cross::SPIRType::Int:	          
			if (type.vecsize == 1)                    return ShaderDataType::Int;
		case spirv_cross::SPIRType::UInt:             return ShaderDataType::Uint;
		case spirv_cross::SPIRType::Float:	          
			switch (type.columns)			          
			{								          
			case 3:							          return ShaderDataType::Mat3;
			case 4:							          return ShaderDataType::Mat4;
			}								          
											          
			switch (type.vecsize)			          
			{								          
			case 1:							          return ShaderDataType::Float;
			case 2:							          return ShaderDataType::Float2;
			case 3:							          return ShaderDataType::Float3;
			case 4:							          return ShaderDataType::Float4;
			}
			break;
		default: ASSERT("Unknown shader data type!"); return ShaderDataType::None;
		}

		ASSERT("unknown type!");
		return ShaderDataType::None;
	}

	static const char* GLShaderStageToString(const GLenum stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:                   return "GL_VERTEX_SHADER";
		case GL_FRAGMENT_SHADER:                 return "GL_FRAGMENT_SHADER";
		default: ASSERT("Unkown shader stage!"); return nullptr;
		}
	}

	static shaderc_shader_kind GLShaderStageToShaderC(const GLenum stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:                    return shaderc_glsl_vertex_shader;
		case GL_FRAGMENT_SHADER:                  return shaderc_glsl_fragment_shader;
		default: ASSERT("Unknown shader stage!"); return shaderc_glsl_default_vertex_shader;
		}
	}

	static const char* GLShaderStageCachedOpenGLFileExtension(const uint32_t stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:                    return ".CachedOpenGL.vert";
		case GL_FRAGMENT_SHADER:                  return ".CachedOpenGL.frag";
		default: ASSERT("Unknown shader stage!"); return nullptr;
		}
	}

	static const char* GLShaderStageCachedVulkanFileExtension(const uint32_t stage)
	{
		switch (stage)
		{
		case GL_VERTEX_SHADER:                    return ".CachedVulkan.vert";
		case GL_FRAGMENT_SHADER:                  return ".CachedVulkan.frag";
		default: ASSERT("Unknown shader stage!"); return nullptr;
		}
	}

	static const char* GetCacheDirectory()
	{
		return "Resources/Shaders/Cache";
	}

	static void CreateCacheDirectoryIfNeeded()
	{
		const std::string cacheDirectory = GetCacheDirectory();
		if (!std::filesystem::exists(cacheDirectory)) {
			std::filesystem::create_directories(cacheDirectory);
		}
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

	Shader::Shader(const std::string& filepath)
		: m_FilePath(filepath)
	{
		CreateCacheDirectoryIfNeeded();

		const std::string source = ReadFile(filepath);
		const auto shaderSources = PreProcess(source);

		CompileOrGetVulkanBinaries(shaderSources);
		CompileOrGetOpenGLBinaries();
		CreateProgram();

		std::cout << "    " << filepath << '\n';

		//for (const auto& v : m_Buffers)
		//{
		//	for(const auto& [key, value] : v.Uniforms)
		//	{
		//		std::cout << "        " << key << '\n';
		//	}
		//}
	}

	Shader::~Shader()
	{
		glDeleteProgram(m_RendererID);
	}

	void Shader::Bind() const
	{
		glUseProgram(m_RendererID);
	}

	void Shader::Unbind()
	{
		glUseProgram(0);
	}

	std::vector<ShaderBuffer>& Shader::GetShaderBuffers()
	{
		return m_Buffers;
	}

	std::string Shader::GetSourceFilepath()
	{
		return m_FilePath;
	}

	uint32_t Shader::GetRendererID() const
	{
		return m_RendererID;
	}

	std::string Shader::ReadFile(const std::string& filepath) const
	{
		std::string result;
		std::ifstream in(filepath, std::ios::in | std::ios::binary);

		if (in)
		{
			in.seekg(0, std::ios::end);
			const size_t size = in.tellg();
			if (size != static_cast<size_t>(-1))
			{
				result.resize(size);
				in.seekg(0, std::ios::beg);
				in.read(result.data(), static_cast<std::streamsize>(size));
			}
			else
			{
				ERR("could not read file (" + filepath + ")");
			}
		}
		else
		{
			ERR("could not open file (" + filepath + ")");
		}

		return result;
	}

	std::unordered_map<GLenum, std::string> Shader::PreProcess(const std::string& source) const
	{
		std::unordered_map<GLenum, std::string> shaderSources;

		const static char* typeToken = "#type";
		const size_t typeTokenLength = strlen(typeToken);
		size_t pos = source.find(typeToken, 0); //Start of shader type declaration line

		while (pos != std::string::npos)
		{
			const size_t eol = source.find_first_of("\r\n", pos); //End of shader type declaration line
			ASSERT(eol != std::string::npos, "syntax error");
			const size_t begin = pos + typeTokenLength + 1; //Start of shader type name (after "#type " keyword)
			std::string type = source.substr(begin, eol - begin);
			ASSERT(ShaderTypeFromString(type), "invalid shader type specified");

			const size_t nextLinePos = source.find_first_not_of("\r\n", eol); //Start of shader code after shader type declaration line
			ASSERT(nextLinePos != std::string::npos, "syntax error");
			pos = source.find(typeToken, nextLinePos); //Start of next shader type declaration line

			shaderSources[ShaderTypeFromString(type)] = (pos == std::string::npos) ? source.substr(nextLinePos) : source.substr(nextLinePos, pos - nextLinePos);
		}

		return shaderSources;
	}

	void Shader::CompileOrGetVulkanBinaries(const std::unordered_map<GLenum, std::string>& shaderSources)
	{
		shaderc::CompileOptions options;
		options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
		options.SetOptimizationLevel(shaderc_optimization_level_performance);
		options.SetGenerateDebugInfo();

		std::filesystem::path cacheDirectory = GetCacheDirectory();

		auto& shaderData = m_VulkanSPIRV;
		shaderData.clear();

		for (auto&& [stage, source] : shaderSources)
		{
			std::filesystem::path shaderFilePath = m_FilePath;
			std::filesystem::path cachedPath = cacheDirectory / (shaderFilePath.filename().string() + GLShaderStageCachedVulkanFileExtension(stage));

			std::ifstream in(cachedPath, std::ios::in | std::ios::binary);

			// Check shader cache
			if (in.is_open())
			{
				in.seekg(0, std::ios::end);
				auto size = in.tellg();
				in.seekg(0, std::ios::beg);

				auto& data = shaderData[stage];
				data.resize(size / sizeof(uint32_t));
				in.read(reinterpret_cast<char*>(data.data()), size);
			}
			// Shader is not cached, cache it
			else
			{
				shaderc::Compiler compiler;
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
					out.write(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(uint32_t)));
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

		shaderc::CompileOptions options;
		options.SetTargetEnvironment(shaderc_target_env_opengl, shaderc_env_version_opengl_4_5);
		options.SetOptimizationLevel(shaderc_optimization_level_performance);
		options.SetGenerateDebugInfo();

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
				in.read(reinterpret_cast<char*>(data.data()), size);
			}
			else
			{
				shaderc::Compiler compiler;
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
					out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(uint32_t)));
					out.flush();
					out.close();
				}
			}
		}
	}

	void Shader::CreateProgram()
	{
		const GLuint program = glCreateProgram();

		std::vector<uint32_t> shaderIDs;
		for (auto&& [stage, spirv] : m_OpenGLSPIRV)
		{
			uint32_t shaderID = shaderIDs.emplace_back(glCreateShader(stage));
			glShaderBinary(1, &shaderID, GL_SHADER_BINARY_FORMAT_SPIR_V, spirv.data(), static_cast<GLsizei>(spirv.size() * sizeof(uint32_t)));
			glSpecializeShader(shaderID, "main", 0, nullptr, nullptr);
			glAttachShader(program, shaderID);
		}

		glLinkProgram(program);
		int isLinked;
		glGetProgramiv(program, GL_LINK_STATUS, &isLinked);

		if (isLinked == GL_FALSE)
		{
			int maxLength;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &maxLength);

			std::vector<char> infoLog(maxLength);
			glGetProgramInfoLog(program, maxLength, &maxLength, infoLog.data());
			ERR("failed to link shader (" + m_FilePath + ")");
			ERR(infoLog.data());

			glDeleteProgram(program);

			for (const auto id : shaderIDs) {
				glDeleteShader(id);
			}
		}

		for (const auto id : shaderIDs)
		{
			glDetachShader(program, id);
			glDeleteShader(id);
		}

		m_RendererID = program;
	}

	ShaderUniform::ShaderUniform(std::string name, ShaderDataType type, uint32_t size, uint32_t offset)
		: m_Name(std::move(name)), m_Type(type), m_Size(size), m_Offset(offset)
	{}

	void Shader::Reflect(GLenum stage, const std::vector<uint32_t>& shaderData)
	{
		const spirv_cross::Compiler compiler(shaderData);
		spirv_cross::ShaderResources resources = compiler.get_shader_resources();
		uint32_t bindingIndex = 0;

		// Extract uniform buffers
		for (const auto& resource : resources.uniform_buffers) {
			auto& bufferType = compiler.get_type(resource.base_type_id);

			const std::string& bufferName = resource.name;
			const auto bufferSize = compiler.get_declared_struct_size(bufferType);

			ShaderBuffer& buffer = m_Buffers.emplace_back();
			buffer.Name = bufferName;
			buffer.Size = static_cast<unsigned>(bufferSize);
			buffer.IsPropertyBuffer = bufferName == "Properties";

			const auto memberCount = bufferType.member_types.size();

			// Member data
			for (uint8_t i = 0; i < memberCount; i++)
			{
				const ShaderDataType uniformType = SPIRTypeToShaderDataType(compiler.get_type(bufferType.member_types[i]));
				const std::string uniformName = compiler.get_member_name(bufferType.self, i);
				const auto uniformSize = compiler.get_declared_struct_member_size(bufferType, i);
				const uint32_t uniformOffset = compiler.type_struct_member_offset(bufferType, i);
				buffer.Uniforms[uniformName] = ShaderUniform(uniformName, uniformType, static_cast<unsigned>(uniformSize), uniformOffset);
			}

			buffer.Buffer = Ref<UniformBuffer>::Create(bufferSize, bindingIndex);
			bindingIndex++;
		}
	}

	std::unordered_map<std::string, Ref<Shader>> ShaderLibrary::m_Shaders;

	Ref<Shader> ShaderLibrary::GetShader(const std::string& filepath)
	{
		if (m_Shaders.contains(filepath)) {
			return m_Shaders[filepath];
		}

		ASSERT("No shader found! (" + filepath + ")");
		return Ref<Shader>();
	}

	void ShaderLibrary::AddShader(const std::string& filepath)
	{
		m_Shaders[filepath] = Ref<Shader>::Create(filepath);
	}

	const std::unordered_map<std::string, Ref<Shader>>& ShaderLibrary::GetShaders()
	{
		return m_Shaders;
	}

	const std::string& ShaderUniform::GetName() const
	{
		return m_Name;
	}

	ShaderDataType ShaderUniform::GetType() const
	{
		return m_Type;
	}

	unsigned int ShaderUniform::GetSize() const
	{
		return m_Size;
	}

	unsigned int ShaderUniform::GetOffset() const
	{
		return m_Offset;
	}
}