// ID qualified

// Vertex program
#type vertex
#version 460 core

layout(location = 0) in vec3 a_Position;

// Data
layout(std140, binding = 0) uniform Data {
	mat4 model;
	mat4 view;
	mat4 proj;
	int id;
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
layout(location = 1) out flat int o_ID;

void main()
{
	gl_Position = proj * view * model * vec4(a_Position, 1);
	Output.Color = vec4(color, 1);
	o_ID = id;
}

// Fragment program
#type fragment
#version 460 core

layout(location = 0) out vec4 o_Color;
layout(location = 1) out int o_ID;

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) in VertexOutput Input;
layout(location = 1) in flat int ID;

void main()
{
	o_Color = Input.Color;
	o_ID = ID;
}