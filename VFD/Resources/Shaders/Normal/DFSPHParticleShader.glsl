// Vertex Shader
#type vertex
#version 450 core

layout(location = 0 ) in vec3  a_Position;
layout(location = 1 ) in vec3  a_Velocity;
layout(location = 2 ) in vec3  a_Acceleration;
layout(location = 3 ) in float a_Mass;
layout(location = 4 ) in float a_Density;
layout(location = 5 ) in float a_Kappa;
layout(location = 6 ) in float a_KappaVelocity;
// Viscosity		  		   
layout(location = 7 ) in vec3  a_ViscosityDifference;
//Surface tension	  		   
layout(location = 8 ) in vec3  a_MonteCarloSurfaceNormals;
layout(location = 9 ) in vec3  a_MonteCarloSurfaceNormalsSmooth;
layout(location = 10) in float a_FinalCurvature;
layout(location = 11) in float a_DeltaFinalCurvature;
layout(location = 12) in float a_SmoothedCurvature;
layout(location = 13) in float a_MonteCarloSurfaceCurvature;
layout(location = 14) in float a_MonteCarloSurfaceCurvatureSmooth;
layout(location = 15) in float a_ClassifierInput;
layout(location = 16) in float a_ClassifierOutput;

layout(std140, binding = 0) uniform Data{
	mat4 view;
	mat4 proj;
	mat4 model;
	vec2 viewportSize;
	float radius;
};

//layout(std140, binding = 1) uniform Properties {
//	vec4 color;
//};

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

	Output.Color = vec4(0.6f, 0.6f, 0.6f, 1.0f);
	Output.Center = (0.5f * gl_Position.xy / gl_Position.w + 0.5f) * viewportSize;
	Output.RadiusInPixels = gl_PointSize / 2.0f;
}

// Fragment Shader
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

	o_Color = vec4(vec3(sqrt(1.52f - l * l)), 1.0f) * Input.Color;
}