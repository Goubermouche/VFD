// Basic flat color shader
// Vertex program
#type vertex
#version 450 core

layout(location = 0) in vec3 a_Position;

layout(std140, binding = 0) uniform Camera
{
	mat4 u_View;
	mat4 u_Projection;
};

layout(std140, binding = 1) uniform Mesh
{
	mat4 u_Model;
	vec4 u_Color;
};

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) out VertexOutput Output;

void main()
{
	Output.Color = u_Color;
	gl_Position = u_Projection * u_View * u_Model * vec4(a_Position, 1.0);
}

// Fragment program
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
