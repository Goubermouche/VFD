#include "pch.h"
#include "RigidBody.h"

namespace vfd
{
	RigidBody::RigidBody(const RigidBodyDescription& desc)
		: m_Description(desc)
	{
		m_Mesh = Ref<TriangleMesh>::Create(desc.SourceMesh);
		m_Data = new RigidBodyDeviceData(desc);
	}

	RigidBody::~RigidBody()
	{
		delete m_Data;
	}
}