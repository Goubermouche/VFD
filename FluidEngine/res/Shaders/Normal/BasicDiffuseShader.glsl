#shader vertex
#version 460 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;

out vec4 frage_Color;
flat out int objectId;

uniform ShaderData{
	uniform vec3 color;
	uniform mat4 model;
	uniform mat4 view;
	uniform mat4 proj;
	uniform int id;
};

void main()
{
	vec3 cameraSpaceVector = normalize((view * model * vec4(a_Normal, 0)).xyz);
	vec3 cameraVector = normalize(vec3(0, 0, 0) - (view * model * vec4(a_Position, 1)).xyz);
	float cosTheta = clamp(dot(cameraSpaceVector, cameraVector), 0, 1);

	gl_Position = proj * view * model * vec4(a_Position, 1);
	frage_Color = vec4(0.9 * color.rgb + cosTheta * color.rgb, 1);
	objectId = id;
}

#shader fragment
#version 460 core

in vec4 frage_Color;
flat in int objectId;

layout(location = 0) out vec4 oute_Color;
layout(location = 1) out int outId;

void main()
{
	oute_Color = frage_Color;
	outId = objectId;
}