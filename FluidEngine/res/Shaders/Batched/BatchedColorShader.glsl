#type vertex
#version 450 core

layout(location = 0) in vec3 a_Position;

layout(std140, binding = 0) uniform ShaderData{
	mat4 view;
	mat4 proj;
	vec4 color;
};

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) out VertexOutput Output;

void main()
{
	Output.Color = color;
	gl_Position = proj * view * vec4(a_Position, 1);
}

#type fragment
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