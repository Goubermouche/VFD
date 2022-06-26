#shader vertex
#version 450 core

layout(location = 0) in vec3 a_Position;

uniform ShaderData{
	uniform mat4 view;
	uniform mat4 proj;
	uniform mat4 model;
	uniform vec2 viewportSize;
	uniform vec4 color;
	uniform float radius;
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
	Output.Center = (0.5 * gl_Position.xy / gl_Position.w + 0.5) * viewportSize;
	Output.RadiusInPixels = gl_PointSize / 2.0;
}

#shader fragment
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

	vec3 pos = vec3(coord, sqrt(1.0 - l * l));
	o_Color = vec4(vec3(pos.z), 1.0) * Input.Color;
}