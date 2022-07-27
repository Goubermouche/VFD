#type vertex
#version 460 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;

layout(std140, binding = 0) uniform ShaderData{
	vec3 color;
	mat4 model;
	mat4 view;
	mat4 proj;
	int id;
};

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) out VertexOutput Output;

void main()
{
	vec3 cameraSpaceVector = normalize((view * model * vec4(a_Normal, 0)).xyz);
	vec3 cameraVector = normalize(vec3(0, 0, 0) - (view * model * vec4(a_Position, 1)).xyz);
	float cosTheta = clamp(dot(cameraSpaceVector, cameraVector), 0, 1);

	gl_Position = proj * view * model * vec4(a_Position, 1);
	Output.Color = vec4(0.9 * color.rgb + cosTheta * color.rgb, 1);
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