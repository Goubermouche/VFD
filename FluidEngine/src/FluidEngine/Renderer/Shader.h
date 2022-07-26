#ifndef SHADER_H_
#define SHADER_H_

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

    struct ShaderProgramSource
    {
        std::string vertexSource;
        std::string fragmentSource;
        std::string geometrySource;
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

    inline std::string ShaderDataTypeNameFromInt(int type) {
        switch (type)
        {
        case 0: return "None";
        case 1: return "Bool";
        case 2: return "Int";
        case 3: return "Uint";
        case 4: return "Float";
        case 5: return "Float2";
        case 6: return "Float3";
        case 7: return "Float4";
        case 8: return "Mat3";
        case 9: return "Mat4";
        }

        ERROR("unknown shader data type '" + std::to_string(type) + "'", "shader");
        return "";
    }

    struct ShaderBuffer {
        std::string Name;
        uint32_t Size = 0;
        std::unordered_map<std::string, ShaderUniform> uniforms;

        void DebugLog() {
            LOG(Name, "uniform buffer");
            for (auto& it : uniforms) {
                LOG("name: " + it.second.GetName());
                LOG("type: " + ShaderDataTypeNameFromInt((int)it.second.GetType()));
                LOG("size: " + std::to_string(it.second.GetSize()));
                LOG("offset: " + std::to_string(it.second.GetOffset()));
                LOG("");
            }
        }
    };

    // Shader uniform naming conventions: 
    // float variable;   - regular glsl variable
    // float s_variable; - serializable variable 
    // float e_variable; - editable variable (also falls under the serializable flag) 

    class Shader : public RefCounted{
    public:
        Shader(const std::string& filepath);
        ~Shader();

        void Bind();
        void Unbind();

        const std::unordered_map<std::string, ShaderBuffer>& GetShaderBuffers() const {
            return m_Buffers;
        }

        const uint32_t GetUniformBuffer() const {
            return m_UniformBuffer;
        }

        inline std::string GetSourceFilepath() {
            return m_Filepath;
        }
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
            ERROR("unknown shader data type '" + std::to_string(type) + "'", "shader");

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

            ERROR("unknown shader data type '" + std::to_string((int)type) + "'", "shader");
            return 0;
        }
    private:
        std::string m_Filepath; // shader source filepath
        uint32_t m_RendererID;
        uint32_t m_UniformBuffer;

        std::unordered_map<std::string, ShaderBuffer> m_Buffers;

        uint32_t m_BlockIndex;

        enum ShaderType {
            None = -1,
            Vertex = 0,
            Fragment = 1,
            Geometry = 2
        };
    };
}

#endif // !SHADER_H_