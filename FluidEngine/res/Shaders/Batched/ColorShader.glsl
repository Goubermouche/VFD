#type vertex
#version 450 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec4 a_Color;

layout(std140, binding = 0) uniform Data{
	mat4 view;
	mat4 proj;
};

layout(location = 0) out vec4 v_Color;

void main()
{
	v_Color = a_Color;
	gl_Position = proj * view * vec4(a_Position, 1);
}

#type fragment
#version 450 core

layout(location = 0) out vec4 o_Color;

layout(location = 0) in vec4 v_Color;

void main()
{
	o_Color = v_Color;
}