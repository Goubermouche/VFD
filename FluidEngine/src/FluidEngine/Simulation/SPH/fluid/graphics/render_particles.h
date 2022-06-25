#ifndef RENDER_PARTICLES_H
#define RENDER_PARTICLES_H
#define NumProg 2
//#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
//#include "renderer/shader.h"
//#include "renderer/components/vao.h"
class ParticleRenderer
{
public:
	ParticleRenderer();  ~ParticleRenderer();

	void display();

	//  set
	void setPositions(float* pos, int nPar) { m_pos = pos;	m_numParticles = nPar; }
	void setVertexBuffer(unsigned int vbo, int nPar) { m_vbo = vbo;	m_numParticles = nPar; }
	void setColorBuffer(unsigned int vbo) { m_colorVbo = vbo; }

	void setFOV(float fov) { m_fov = fov;  updPScale(); }
	void setWindowSize(int w, int h) { m_window_w = w;  m_window_h = h;  updPScale(); }
	void updPScale() { m_ParScale = m_window_h / tanf(m_fov * 0.5f * PI / 180.0f); }

protected:  // methods

	void _initGL();
	void _drawPoints();
	GLuint _compileProgram(const char* vsource, const char* fsource);

protected:  // data
public:
	int m_nProg;

	float* m_pos;	
	GLuint m_program[NumProg], m_vbo, m_colorVbo, m_scaleProg;
	int m_numParticles, m_window_w, m_window_h;  float m_fov;

	float m_ParRadius, m_ParScale, m_fDiffuse, m_fAmbient, m_fPower, m_fSteps, m_fHueDiff;
	GLint m_uLocPRadius[NumProg], m_uLocPScale[NumProg], m_uLocDiffuse, m_uLocAmbient, m_uLocPower, m_uLocSteps, m_uLocStepsS, m_uLocHueDiff;

	static const char* vertexShader;
	static const char* spherePixelShader[NumProg];
	static const char* scalePixelShader;
};

#endif // !RENDER_PARTICLES_H


