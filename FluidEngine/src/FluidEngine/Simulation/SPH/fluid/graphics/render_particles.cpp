#include "pch.h"
#include "../pch/header.h"
#include "render_particles.h"



ParticleRenderer::ParticleRenderer() :
	m_pos(0), m_numParticles(0), m_ParRadius(0.04f), m_ParScale(1.0f),
	m_fDiffuse(0.3f), m_fAmbient(0.7f), m_fPower(1.0f), m_fSteps(0), m_fHueDiff(0.0f),
	m_vbo(0), m_colorVbo(0)
{
	m_nProg = 0/*0*/;  m_program[0] = 0;  _initGL();
}

ParticleRenderer::~ParticleRenderer() { m_pos = 0; }

void ParticleRenderer::_drawPoints()
{
	if (!m_vbo)
	{
		/*glBegin(GL_POINTS);  int a = 0;
		for (int i = 0; i < m_numParticles; ++i, a += 4)	glVertex3fv(&m_pos[a]);
		glEnd();*/
		printf("no vbo");
	}
	else
	{
		//

		//glEnableClientState(GL_VERTEX_ARRAY);
		//glVertexPointer(4, GL_FLOAT, 0, 0);

		//glDrawArrays(GL_POINTS, 0, (GLsizei)m_numParticles);

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

		glEnableVertexAttribArray(0);

		glVertexAttribPointer(0, 4, GL_FLOAT, false, 4, 0);

		glDrawElements(GL_POINTS, m_numParticles, GL_FLOAT, NULL);

		glBindBuffer(GL_ARRAY_BUFFER, 0);


	}
}

void ParticleRenderer::display()
{
	/*glEnable(GL_POINT_SPRITE_ARB);
	glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
	glDepthMask(GL_TRUE);	
	glEnable(GL_DEPTH_TEST);*/

	int i = m_nProg;
	//glUseProgram(m_program[i]);	//  pass vars
	//glUniform1f(m_uLocPScale[i], m_ParScale);
	//glUniform1f(m_uLocPRadius[i], m_ParRadius);
	//if (i == 0) {
	//	glUniform1f(m_uLocDiffuse, m_fDiffuse);
	//	glUniform1f(m_uLocAmbient, m_fAmbient);
	//	glUniform1f(m_uLocPower, m_fPower);
	//}
	//else {
	//	glUniform1f(m_uLocHueDiff, m_fHueDiff);
	//	glUniform1f(m_uLocSteps, m_fSteps);
	//	/*glUniform1f( m_uLocStepsS, m_fSteps );*/
	//}

	//glColor3f(1, 1, 1);
	// _drawPoints();

	//glUseProgram(0);
	//glDisable(GL_POINT_SPRITE_ARB);
}

GLuint ParticleRenderer::_compileProgram(const char* vsource, const char* fsource)
{
	GLuint vertexShader;
	if (vsource) {
		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vsource, 0);
		glCompileShader(vertexShader);
	}

	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fsource, 0);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	if (vsource)	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success) {
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);	program = 0;
	}
	else {
		printf("compiled successfully\n");
	}
	return program;
}

void ParticleRenderer::_initGL()
{
	m_scaleProg = _compileProgram(NULL, scalePixelShader);

	for (int i = 0; i < NumProg; i++) {
		m_program[i] = _compileProgram(vertexShader, spherePixelShader[i]);

		//  vars loc
		m_uLocPScale[i] = glGetUniformLocation(m_program[i], "pointScale");
		m_uLocPRadius[i] = glGetUniformLocation(m_program[i], "pointRadius");
	}
	m_uLocHueDiff = glGetUniformLocation(m_program[1], "fHueDiff");
	m_uLocDiffuse = glGetUniformLocation(m_program[0], "fDiffuse");
	m_uLocAmbient = glGetUniformLocation(m_program[0], "fAmbient");
	m_uLocPower = glGetUniformLocation(m_program[0], "fPower");

	m_uLocSteps = glGetUniformLocation(m_program[1], "fSteps");
	m_uLocStepsS = glGetUniformLocation(m_scaleProg, "fSteps");

	//glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
	//glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
}
#define STRING(A) #A
#define STRING2(A,B) #A#B


///  Vertex shader

const char* ParticleRenderer::vertexShader =
STRING(
	uniform float  pointRadius;  // point size in world space
uniform float  pointScale;   // scale to calculate size in pixels

void main()
{
	// calculate window-space point size
	vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
	float dist = length(posEye);
	gl_PointSize = pointRadius * (pointScale / dist);

	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	gl_FrontColor = gl_Color;
}
);


#define HUE2RGB	\
	"#version 120\n\
	\
	vec4 hsv2rgb(vec3 hsv)\n\
	{\n\
		vec3 rgb;\n\
		if (hsv.g == 0.0f)  {\n\
		  if (hsv.b != 0.0f)\n\
				rgb.x = hsv.b;  }\n\
		else\n\
		{\n\
			float h = hsv.r * 6.f;		float s = hsv.g;	float v = hsv.b;\n\
			if (h >= 6.f)	h = 0.f;\n\
			\
			int i = int(floor(h));	float f = h - i;\n\
			\
			float a = 1.f - s;\n\
			float b = 1.f - s * f;\n\
			float c = 1.f - s * (1.f - f);\n\
			\
			/*if (i & 1)  c = b;\n\
			rgb[ap[i]] = a;  rgb[vp[i]] = 1;  rgb[cp[i]] = c;*/\n\
			\
			switch (i)	{\n\
				case 0:  rgb[0] = 1;  rgb[1] = c;  rgb[2] = a;	break;\n\
				case 1:  rgb[0] = b;  rgb[1] = 1;  rgb[2] = a;	break;\n\
				case 2:  rgb[0] = a;  rgb[1] = 1;  rgb[2] = c;	break;\n\
				case 3:  rgb[0] = a;  rgb[1] = b;  rgb[2] = 1;	break;\n\
				case 4:  rgb[0] = c;  rgb[1] = a;  rgb[2] = 1;	break;\n\
				case 5:  rgb[0] = 1;  rgb[1] = a;  rgb[2] = b;	break;	}\n\
			/*rgb *= v;*/\n\
		}\n\
		return vec4(rgb, 1.f);\n\
	}"


///  Pixel shader  for rendering points as shaded spheres

const char* ParticleRenderer::spherePixelShader[NumProg] = {

STRING(		///  Diffuse

	uniform float  fAmbient;   // factors
	uniform float  fDiffuse;
	uniform float  fPower;

	void main()
	{
		//const vec3 lightDir = vec3(0.577, 0.577, 0.577);

		//// calculate normal from texture coordinates
		//vec3 n;  n.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
		//float mag = dot(n.xy, n.xy);
		//if (mag > 1.0)  discard;   // don't draw outside circle
		//n.z = sqrt(1.0 - mag);

		//// calculate lighting
		//float diffuse = max(0.0, dot(lightDir, n));

		////gl_FragColor = gl_Color*(fAmbient + fDiffuse * diffuse);
		gl_FragColor = gl_Color;
	}


),HUE2RGB STRING(	///  Hue
	uniform float fSteps = 0.f;
	uniform float fHueDiff;

	void main()
	{
		//vec3 n;  n.xy = gl_TexCoord[0].xy * vec2(2,-2) + vec2(-1,1);
		//float mag = dot(n, n);
		//if (mag > 1.0)  discard;   // circle

		//// calculate lighting
		//const vec3 lightDir = vec3(0.577, 0.577, 0.577);
		//n.z = sqrt(1.0 - mag);
		//float diffuse = max(0.0, dot(lightDir, n));

		//n.x = gl_Color.r;
		//if (fSteps > 0.f) { int i = n.x * fSteps;  n.x = i / fSteps; }
		//float h = 0.83333f - n.x;
		//float s = 1.f;
		//if (h < 0.f) { s += h * 6;  h = 0.f; }
		////gl_FragColor = hsv2rgb(vec3(h, s - gl_Color.g/*dye*/, 1.f));

		gl_FragColor = vec4(0, 1, 0, 1);
	}

) };


const char* ParticleRenderer::scalePixelShader =
HUE2RGB STRING(		///  Hue Scale
	\n uniform float fSteps = 0.f; \n

	//uniform float  fBright;\n
	//uniform float  fContrast;\n

	void main()\n
	{ \n
	//vec2 n;  n = gl_TexCoord[0].xy; \n

	//if (fSteps > 0.f) { int i = n.x * fSteps;  n.x = i / fSteps; }\n
	//float h = 0.83333f - int(n.x); \n
	//float s = 1.f; \n
	//if (h < 0.f) { s += h * 6;  h = 0.f; }\n
	gl_FragColor = vec4(1, 0, 0, 1); \n
}\n

);
