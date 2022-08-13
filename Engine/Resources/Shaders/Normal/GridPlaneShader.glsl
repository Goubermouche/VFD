// Vertex Shader
#type vertex
#version 460 core

layout(location = 0) in vec3 a_Position;

layout(std140, binding = 0) uniform Data{
	mat4 view;
	mat4 proj;
    vec3 color;
    float near;
    float far;
    float scale;
};

struct VertexOutput {
    mat4 FragView;
    mat4 FragProj;
    vec3 Color;
    vec3 NearPoint;
    vec3 FarPoint;
    float Near;
    float Far;
    float Scale;
};

layout(location = 0) out VertexOutput Output;

vec3 UnprojectPoint(float x, float y, float z, mat4 view, mat4 projection) {
	mat4 viewInv = inverse(view);
	mat4 projInv = inverse(projection);
	vec4 unprojectedPoint = viewInv * projInv * vec4(x, y, z, 1.0);
	return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main()
{
	Output.NearPoint = UnprojectPoint(a_Position.x, a_Position.y, 0.0, view, proj).xyz;
    Output.FarPoint = UnprojectPoint(a_Position.x, a_Position.y, 1.0, view, proj).xyz;
    Output.FragView = view;
    Output.FragProj = proj;
    Output.Color = color;
    Output.Scale = scale;
    Output.Near = near;
    Output.Far = far;

	gl_Position = vec4(a_Position, 1.0);
}

// Fragment Shader
#type fragment
#version 460 core

layout(location = 0) out vec4 o_Color;

struct VertexOutput {
    mat4 FragView;
    mat4 FragProj;
    vec3 Color;
    vec3 NearPoint;
    vec3 FarPoint;
    float Near;
    float Far;
    float Scale;
};

layout(location = 0) in VertexOutput Input;

vec4 Grid(vec3 fragPos3D, float scale) {
    vec2 coord = fragPos3D.xz * scale;
    vec2 derivative = fwidth(coord);
    vec2 grid = abs(fract(coord - 0.5) - 0.5) / derivative;
    float line = min(grid.x, grid.y);
    return vec4(Input.Color.xyz, 1.0 - min(line, 1.0));
}

void main() {
    float t = -Input.NearPoint.y / (Input.FarPoint.y - Input.NearPoint.y);
    vec3 fragPos3D = Input.NearPoint + t * (Input.FarPoint - Input.NearPoint);
    vec4 clipSpacePos = Input.FragProj * Input.FragView * vec4(fragPos3D.xyz, 1.0);

    // Scene depth
    gl_FragDepth = 0.5 + 0.5 * (clipSpacePos.z / clipSpacePos.w);

    // Linear depth
    float clipSpaceDepth = (clipSpacePos.z / clipSpacePos.w) * 2.0 - 1.0;
    float linearDepth = ((2.0 * Input.Near * Input.Far) / (Input.Far + Input.Near - clipSpaceDepth * (Input.Far - Input.Near))) / Input.Far;
    float fading = max(0, (0.5 - linearDepth));

    // small grid + large grid
    o_Color = (Grid(fragPos3D, Input.Scale) + Grid(fragPos3D, Input.Scale * 10)) * float(t > 0);
    o_Color.a *= max(0, (0.5 - linearDepth));
}