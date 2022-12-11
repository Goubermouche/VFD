#include "pch.h"
#include "GPUDFSPHSimulator.h"
#include "Renderer/Renderer.h"
#include "Debug/SystemInfo.h"

namespace vfd
{
	GPUDFSPHSimulation::GPUDFSPHSimulation(const GPUDFSPHSimulationDescription& desc)
		: m_Description(desc)
	{
		if (SystemInfo::CUDADeviceMeetsRequirements() == false) {
			return;
		}

		// Init rigidbodies 
		{
			RigidBodyDescription rigidbodyDesc;

			glm::mat4 transform(1.0f);

			transform = glm::translate(transform, { 0.0f, -2.0, 0.0f });
			transform = glm::rotate(transform, glm::radians(45.0f), { 1.0f, 0.0f, 0.0f });
			transform = glm::scale(transform, { 1.0f, 0.1f, 1.0f });

			rigidbodyDesc.Transform = transform;
			rigidbodyDesc.CollisionMapResolution = { 10, 10, 10 };
			rigidbodyDesc.SourceMesh = "Resources/Models/Cube.obj";
			rigidbodyDesc.Inverted = false;
			rigidbodyDesc.Padding = 0.0f;

			const Ref<RigidBody> rigidbody = Ref<RigidBody>::Create(rigidbodyDesc);
			m_RigidBodies.push_back(rigidbody);
		}

		//{
		//	RigidBodyDescription rigidbodyDesc;

		//	glm::mat4 transform(1.0f);

		//	transform = glm::translate(transform, { 0.0f, 0.0f, 0.0f });
		//	transform = glm::rotate(transform, glm::radians(0.0f), { 1.0f, 0.0f, 0.0f });
		//	transform = glm::scale(transform, { 1.0f, 1.0f, 1.0f });

		//	rigidbodyDesc.Transform = transform;
		//	rigidbodyDesc.CollisionMapResolution = { 10, 10, 10 };
		//	rigidbodyDesc.SourceMesh = "Resources/Models/Torus.obj";
		//	rigidbodyDesc.Inverted = false;
		//	rigidbodyDesc.Padding = 0.0f;

		//	const Ref<RigidBody> rigidbody = Ref<RigidBody>::Create(rigidbodyDesc);
		//	m_RigidBodies.push_back(rigidbody);
		//}

		m_RigidBodyMaterial = Ref<Material>::Create(Renderer::GetShader("Resources/Shaders/Normal/BasicDiffuseShader.glsl"));
		m_RigidBodyMaterial->Set("color", { 0.4f, 0.4f, 0.4f, 1 });

		m_Implementation = Ref<DFSPHImplementation>::Create();
	}

	const Ref<VertexArray>& GPUDFSPHSimulation::GetVertexArray()
	{
		return m_Implementation->GetVertexArray();
	}

	void GPUDFSPHSimulation::OnUpdate()
	{
		if (paused) {
			return;
		}

		m_Implementation->OnUpdate();
	}
}