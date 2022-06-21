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
};

layout(location = 0) out VertexOutput Output;

const float radiusFactor = 30;

void main()
{
	Output.Color = color;
	gl_Position = proj * view * model * vec4(a_Position, 1);
	gl_PointSize = viewportSize.y * proj[1][1] * radius / radiusFactor / gl_Position.w;
}

#shader fragment
#version 450 core

layout(location = 0) out vec4 o_Color;

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) in VertexOutput Input;

void main()
{
	vec3 n;
	n.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(n.xy, n.xy);

	if (mag > 1.0) {
		discard;
	}

	o_Color = Input.Color;
}