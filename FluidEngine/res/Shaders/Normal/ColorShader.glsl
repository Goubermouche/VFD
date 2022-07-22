#shader vertex
#version 460 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

out vec4 fragColor;

uniform ShaderData{
	uniform vec3 color;
	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 proj;
};

void main()
{
	gl_Position = proj * view * model * vec4(aPos, 1);
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