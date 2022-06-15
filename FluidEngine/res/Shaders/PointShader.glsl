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
};

layout(location = 0) out VertexOutput Output;

const float radiusFactor = 30;

void main()
{
	Output.Color = a_Color;
	gl_Position = proj * view * vec4(a_Position, 1);
	gl_PointSize = viewportSize.y * proj[1][1] * a_Radius / radiusFactor / gl_Position.w;
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