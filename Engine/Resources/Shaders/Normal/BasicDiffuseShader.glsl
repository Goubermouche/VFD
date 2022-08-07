// Simple diffuse shader
// Vertex program
#type vertex
#version 460 core

layout(location = 0) in vec3 a_Position;
layout(location = 1) in vec3 a_Normal;

// Data
layout(std140, binding = 0) uniform Data
{
	mat4 model;
	mat4 view;
	mat4 proj;
	int entityID;
};

// Properties
layout(std140, binding = 1) uniform Properties
{
	vec4 color;
};

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) out VertexOutput Output;
layout(location = 1) out flat int ID;

void main()
{
	vec3 cameraSpaceVector = normalize((view * model * vec4(a_Normal, 0)).xyz);
	vec3 cameraVector = normalize(vec3(0, 0, 0) - (view * model * vec4(a_Position, 1)).xyz);
	float cosTheta = clamp(dot(cameraSpaceVector, cameraVector), 0, 1);

	gl_Position = proj * view * model * vec4(a_Position, 1);

	Output.Color = vec4(0.9 * color.rgb + cosTheta * color.rgb, 1);
	ID = entityID;
}

// Fragment program
#type fragment
#version 460 core

layout(location = 0) out vec4 o_Color;
layout(location = 1) out int o_ID;

struct VertexOutput
{
	vec4 Color;
};

layout(location = 0) in VertexOutput Input;
layout(location = 1) in flat int ID;


void main()
{
	o_Color = Input.Color;
	o_ID = ID;
}