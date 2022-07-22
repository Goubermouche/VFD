#shader vertex
#version 450 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec4 a_Color;
layout(location = 2) in float a_Radius;

uniform ShaderData{
	uniform mat4 view;
	uniform mat4 proj;
	uniform vec2 viewportSize;
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
	gl_Position = proj * view * vec4(a_Position, 1);
	gl_PointSize = viewportSize.y * proj[1][1] * a_Radius / radiusFactor / gl_Position.w;

	Output.Color = a_Color;
	Output.Center = (0.5 * gl_Position.xy / gl_Position.w + 0.5) * viewportSize;
	Output.RadiusInPixels = gl_PointSize / 2.0;
}

#shader fragment
#version 450 core

layout(location = 0) out vec4 o_Color;

struct VertexInput
{
	vec4 Color;
	vec2 Center;
	float RadiusInPixels;
};

layout(location = 0) in VertexInput Input;

void main()
{
	vec2 coord = (gl_FragCoord.xy - Input.Center) / Input.RadiusInPixels;
	float l = length(coord);
	if (l > 1.0) {
		discard;
	}

	vec3 pos = vec3(coord, sqrt(1.0 - l * l));

	o_Color = vec4(vec3(pos.z), 1.0) * Input.Color;
}