#include "pch.h"
#include "SPHSimulation.h"

namespace fe {
	const float scale = 10.0f;;
	
	SPHSimulation::SPHSimulation()
	{
		m_PointMaterial = Material::Create(Shader::Create("res/Shaders/Normal/PointColorShader.glsl"));
		m_PointMaterial->Set("color", { 0.271,1.,0.757, 1 });
		m_PointMaterial->Set("radius", 0.8f);
		m_PointMaterial->Set("model", glm::scale(glm::mat4(1.0f), { scale, scale , scale }));
	}

	SPHSimulation::~SPHSimulation()
	{
	}

	void SPHSimulation::OnUpdate()
	{
	}

	void SPHSimulation::OnRender()
	{
		Renderer::DrawLine({ 0, 0, 0 }, { 10, 10, 10 }, { 1, 1,0,1 });
	}
}