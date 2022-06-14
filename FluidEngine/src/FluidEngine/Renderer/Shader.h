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

        const std::string& GetName() const { return m_Name; }
        ShaderDataType GetType() const { return m_Type; }
        uint32_t GetSize() const { return m_Size; }
        uint32_t GetOffset() const { return m_Offset; }

        static constexpr std::string_view UniformTypeToString(ShaderDataType type);

    private:
        std::string m_Name;
        ShaderDataType m_Type = ShaderDataType::None;
        uint32_t m_Size = 0;
        uint32_t m_Offset = 0;
    };

    // CHECK: 
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

        return "Unknown";
    }

    struct ShaderBuffer {
        std::string Name;
        uint32_t Size = 0;
        std::unordered_map<std::string, ShaderUniform> uniforms;

        void Log() {
            LOG(Name);
            for (auto& it : uniforms) {
                std::cout << "    Name:   " << it.second.GetName() << std::endl;
                std::cout << "    Type:   " << ShaderDataTypeNameFromInt((int)it.second.GetType()) << std::endl;
                std::cout << "    Size:   " << it.second.GetSize() << std::endl;
                std::cout << "    Offset: " << it.second.GetOffset() << std::endl << std::endl;
            }
        }
    };

    class Shader : public RefCounted{
    public:
        virtual ~Shader() {};

        virtual void Bind() = 0;
        virtual void Unbind() = 0;

        virtual const uint32_t GetUniformBuffer() const = 0;
        virtual const std::unordered_map<std::string, ShaderBuffer>& GetShaderBuffers() const = 0;

        static Ref<Shader> Create(const std::string& filePath);
    };
}

#endif // !SHADER_H_