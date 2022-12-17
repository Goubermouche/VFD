#ifndef  RIGID_BODY_OBJECT_H
#define  RIGID_BODY_OBJECT_H

#include "RigidBodyData.h"
#include "Renderer/Mesh/TriangleMesh.h"
#include "RigidBodyImplementation.h"

namespace vfd
{
	struct RigidBodyDescription
	{
		glm::mat4 Transform; // TODO: Use the transform component
		glm::uvec3 CollisionMapResolution = { 10, 10, 10 };
		std::string SourceMesh;

		bool Inverted;
		float Padding;
	};

	class RigidBody : public RefCounted
	{
	public:
		RigidBody(const RigidBodyDescription& desc);

		RigidBodyData* GetData()
		{
			return m_Data;
		}

		Ref<TriangleMesh>& GetMesh()&
		{
			return m_Mesh;
		}

		const glm::mat4& GetTransform() const
		{
			return m_Description.Transform;
		}

		RigidBodyImplementation* Implementation;
	private:
		RigidBodyDescription m_Description;
		RigidBodyData* m_Data;
		Ref<TriangleMesh> m_Mesh;
	};
}

#endif // !RIGID_BODY_OBJECT_H