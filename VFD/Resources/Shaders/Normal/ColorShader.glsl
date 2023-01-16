// Vertex program
#type vertex
#version 460 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;

// Data
layout(std140, binding = 0) uniform Data
{
	mat4 model;
	mat4 view;
	mat4 proj;
};

// Properties
layout(std140, binding = 1) uniform Properties
{
	vec4 color;
};

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) out VertexOutput Output;

void main()
{
	gl_Position = proj * view * model * vec4(a_Position, 1);

	Output.Color = color;
}

// Fragment program
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