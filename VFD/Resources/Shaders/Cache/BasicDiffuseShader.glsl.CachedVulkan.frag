#  
                   GLSL.std.450              	       main    
                         Resources/Shaders/Normal/BasicDiffuseShader.glsl     K    �     #version 460 core

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
}  
 GL_GOOGLE_cpp_style_line_directive    GL_GOOGLE_include_directive      main      
   o_Color      VertexOutput             Color        Input        o_ID         ID  J entry-point main    J client vulkan100    J target-env spirv1.5 J target-env vulkan1.2    J entry-point main    G  
          G            G           G        G                !                               	         ;  	   
                          ;                       +                                    ;                       ;                      6               �                 A              =           >  
                  =           >        �  8  