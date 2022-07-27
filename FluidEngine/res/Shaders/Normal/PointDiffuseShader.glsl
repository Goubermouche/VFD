#type vertex
#version 450 core

layout(location = 0) in vec3 a_Position;

layout(std140, binding = 0) uniform ShaderData{
	mat4 view;
	mat4 proj;
	mat4 model;
	vec2 viewportSize;
	vec4 color;
	float radius;
};

struct VertexOutput
{
	vec4 Color;
	vec2 Center;
	float RadiusInPixels;
};

layout(location = 0) out VertexOutput Output;

const float radiusFactor = 30;

void main()
{
	gl_Position = proj * view * model * vec4(a_Position, 1);
	gl_PointSize = viewportSize.y * proj[1][1] * radius / radiusFactor / gl_Position.w;

	Output.Color = color;
	Output.Center = (0.5f * gl_Position.xy / gl_Position.w + 0.5f) * viewportSize;
	Output.RadiusInPixels = gl_PointSize / 2.0f;
}

#type fragment
#version 450 core

layout(location = 0) out vec4 o_Color;

struct VertexOutput
{
	vec4 Color;
	vec2 Center;
	float RadiusInPixels;
};

layout(location = 0) in VertexOutput Input;

void main()
{
	vec2 coord = (gl_FragCoord.xy - Input.Center) / Input.RadiusInPixels;
	float l = length(coord);
	if (l > 1.0) {
		discard;
	}

	o_Color = vec4(vec3(sqrt(1.2f - l * l)), 1.0f) * Input.Color;
}