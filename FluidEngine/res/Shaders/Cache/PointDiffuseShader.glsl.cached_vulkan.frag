#  
  =                 GLSL.std.450                     main          +                res/Shaders/Normal/PointDiffuseShader.glsl   k    �     #version 450 core

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

	o_Color = vec4(vec3(sqrt(1.2f - l * l)), 1.0f) * Input.Color;
}     
 GL_GOOGLE_cpp_style_line_directive    GL_GOOGLE_include_directive      main         gl_FragCoord         VertexOutput             Color           Center          RadiusInPixels       Input     +   o_Color J entry-point main    J client vulkan100    J target-env spirv1.5 J target-env vulkan1.2    J entry-point main    G           G            G  +               !                                                   ;                                     ;                       +                       +                       +     $     �?  %      *         ;  *   +      +     ,   ���?+     8                  6               �                 =           O                     A              =           �              A              =           P              �                               "      B                  �  %   &   "   $   �  (       �  &   '   (   �  '               �  �  (        <   "                    0      2   <   "   ,        1         0   P     7   1   1   1   $   A     9      8   =     :   9   �     ;   7   :   >  +   ;   �  8  