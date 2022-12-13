#include "pch.h"
#include "RigidBodyObject.h"

namespace vfd
{
	RigidBody::RigidBody(const RigidBodyDescription& desc)
		: m_Description(desc)
	{
		m_Mesh = Ref<TriangleMesh>::Create(desc.SourceMesh);
		m_Data = new RigidBodyData(m_Description);
	}
}