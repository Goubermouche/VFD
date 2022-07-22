#shader vertex
#version 450 core

layout(location = 0) in vec3 a_Position;

uniform ShaderData{
	uniform mat4 view;
	uniform mat4 proj;
};

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) out VertexOutput Output;

void main()
{
	Output.Color = glm::vec4(1, 0, 0, 1);
	gl_Position = proj * view * vec4(a_Position, 1);
}

#shader fragment
#version 450 core

layout(location = 0) out vec4 o_Color;

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) in VertexOutput Input;

void main()
{
	o_Color = Input.Color;
}