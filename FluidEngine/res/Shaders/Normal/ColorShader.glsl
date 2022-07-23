#shader vertex
#version 460 core

layout(location = 0) in vec3 a_Position;

out vec4 fragColor;

uniform ShaderData{
	uniform vec3 color;
	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 proj;
};

void main()
{
	gl_Position = proj * view * model * vec4(a_Position, 1);
	fragColor = vec4(color, 1);
}

#shader fragment
#version 460 core

in vec4 fragColor;

layout(location = 0) out vec4 outColor;

void main()
{
	outColor = fragColor;
}