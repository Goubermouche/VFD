#type vertex
#version 460 core

layout(location = 0) in vec3 a_Position;

layout(std140, binding = 0) uniform Data{
	vec3 color;
	mat4 model;
	mat4 view;
	mat4 proj;
};

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) out VertexOutput Output;

void main()
{
	gl_Position = proj * view * model * vec4(a_Position, 1);
	Output.Color = vec4(color, 1);
}

#type fragment
#version 460 core

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