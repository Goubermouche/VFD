#include "pch.h"
#include "Shader.h"

#include "FluidEngine/Utility/FileSystem.h"
#include <Glad/glad.h>

namespace fe {
	ShaderUniform::ShaderUniform(std::string name, ShaderDataType type, uint32_t size, uint32_t offset)
		: m_Name(std::move(name)), m_Type(type), m_Size(size), m_Offset(offset)
	{}

	Shader::Shader(const std::string& filepath)
		: m_Filepath(filepath), m_RendererID(0)
	{
		ASSERT(FileExists(filepath), "file path is invalid! (" + filepath + ")");

		// TODO: this section is currently a bit cursed, clean it up and find out how to use textures with it
		const ShaderProgramSource source = Parse(filepath);
		m_RendererID = CreateProgram(source.vertexSource, source.fragmentSource, source.geometrySource);

		GLint numBuffers;
		glGetProgramInterfaceiv(m_RendererID, GL_UNIFORM_BLOCK, GL_ACTIVE_RESOURCES, &numBuffers);

		std::vector<GLenum> bufferProperties;
		bufferProperties.push_back(GL_NAME_LENGTH);
		bufferProperties.push_back(GL_BUFFER_DATA_SIZE);
		bufferProperties.push_back(GL_NUM_ACTIVE_VARIABLES);
		std::vector<GLint> bufferValues(bufferProperties.size());

		std::vector<GLenum> uniformProperties;
		uniformProperties.push_back(GL_NAME_LENGTH);
		uniformProperties.push_back(GL_TYPE);
		uniformProperties.push_back(GL_OFFSET);
		uniformProperties.push_back(GL_BLOCK_INDEX);
		std::vector<GLint> uniformValues(uniformProperties.size());

		std::vector<GLchar> nameData(256);
		uint32_t uniformIndex = 0;

		// TODO: add support for multiple buffers
		for (size_t i = 0; i < numBuffers; i++)
		{
			glGetProgramResourceiv(m_RendererID, GL_UNIFORM_BLOCK, i, bufferProperties.size(), &bufferProperties[0], bufferValues.size(), NULL, &bufferValues[0]);
			nameData.resize(bufferValues[0]);
			glGetProgramResourceName(m_RendererID, GL_UNIFORM_BLOCK, i, nameData.size(), NULL, &nameData[0]);
			std::string bufferName((char*)&nameData[0], nameData.size() - 1);
			ShaderBuffer& buffer = m_Buffers[bufferName];
			buffer.Name = bufferName;
			buffer.Size = bufferValues[1];
			uint32_t memberCount = bufferValues[2];

			for (size_t j = 0; j < memberCount; j++)
			{
				glGetProgramResourceiv(m_RendererID, GL_UNIFORM, j, uniformProperties.size(), &uniformProperties[0], uniformValues.size(), NULL, &uniformValues[0]);

				// temp solution 
				if (uniformValues[3] == -1) {
					memberCount++;
					continue;
				}

				nameData.resize(uniformValues[0]);
				glGetProgramResourceName(m_RendererID, GL_UNIFORM, j, nameData.size(), NULL, &nameData[0]);
				auto type = GetShaderDataTypeFromGLenum(uniformValues[1]);
				std::string uniformName((char*)&nameData[0], nameData.size() - 1);
				auto size = GetShaderDataTypeSize(type);
				auto offset = uniformValues[2];
				buffer.uniforms[uniformName] = ShaderUniform(uniformName, type, size, offset);
			}

			// buffer.DebugLog();
		}

		glGenBuffers(1, &m_UniformBuffer);
	}

	Shader::~Shader()
	{
		glDeleteProgram(m_RendererID);
	}

	void Shader::Bind()
	{
		glUseProgram(m_RendererID);
	}

	void Shader::Unbind()
	{
		glUseProgram(0);
	}

	ShaderProgramSource Shader::Parse(const std::string& filePath) const
	{
		std::ifstream stream(filePath);
		std::string line;
		std::stringstream ss[3];
		ShaderType type = None;

		while (getline(stream, line))
		{
			if (line.find("#shader") != std::string::npos)
			{

				if (line.find("vertex") != std::string::npos) {
					type = ShaderType::Vertex;
				}
				else if (line.find("fragment") != std::string::npos) {
					type = ShaderType::Fragment;
				}
				else if (line.find("geometry") != std::string::npos) {
					type = ShaderType::Geometry;
				}
			}
			else
			{
				ss[(int)type] << line << '\n';
			}
		}

		ShaderProgramSource programSource = { ss[0].str(), ss[1].str(), ss[2].str() };
		return programSource;
	}

	uint32_t Shader::Compile(const uint32_t type, const std::string& source) const
	{
		const uint32_t id = glCreateShader(type);
		const char* src = source.c_str();
		glShaderSource(id, 1, &src, nullptr);
		glCompileShader(id);

		// Error handling
		int result;
		glGetShaderiv(id, GL_COMPILE_STATUS, &result);

		if (result == GL_FALSE)
		{
			int length;
			glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
			char* message = (char*)alloca(length * sizeof(char));
			glGetShaderInfoLog(id, length, &length, message);
			const std::string shader_type = (type == GL_VERTEX_SHADER) ? "vertex" : "fragment";
			ERR(" failed to compile " + shader_type + " shader");

			std::vector<std::string> strings;
			std::string str = message;
			std::string::size_type pos = 0;
			std::string::size_type prev = 0;

			while ((pos = str.find("\n", prev)) != std::string::npos)
			{
				std::cout << "p" + str.substr(prev, pos - prev) << std::endl;
				ERR(" p" + str.substr(prev, pos - prev));
				prev = pos + 1;
			}

			glDeleteShader(id);
			return 0;
		}

		return id;
	}

	uint32_t Shader::CreateProgram(const std::string& vertexShader, const std::string& fragmentShader, const std::string& geometryShader) const
	{
		// create a shader program
		const unsigned int program = glCreateProgram();
		const unsigned int vs = Compile(GL_VERTEX_SHADER, vertexShader);
		const unsigned int fs = Compile(GL_FRAGMENT_SHADER, fragmentShader);
		unsigned int gs;

		if (!geometryShader.empty()) {
			gs = Compile(GL_GEOMETRY_SHADER, geometryShader);
		}

		glAttachShader(program, vs);
		glAttachShader(program, fs);

		if (!geometryShader.empty()) {
			glAttachShader(program, gs);
		}

		glLinkProgram(program);
		GLint program_linked;
		glGetProgramiv(program, GL_LINK_STATUS, &program_linked);

		if (program_linked != GL_TRUE)
		{
			GLsizei log_length = 0;
			GLchar message[1024];
			glGetProgramInfoLog(program, 1024, &log_length, message);
			ERR(" '" + m_Filepath + "' failed to link");
			ERR(message);
		}
		else {
			LOG("'" + m_Filepath + "' linked successfully", "renderer][shader", ConsoleColor::Green);
		}

		glValidateProgram(program);

		glDeleteShader(vs);
		glDeleteShader(fs);
		if (!geometryShader.empty()) {
			glDeleteShader(gs);
		}

		return program;
	}
}