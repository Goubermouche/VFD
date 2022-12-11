#ifndef RIGID_BODY_IMPLEMENTATION_H
#define RIGID_BODY_IMPLEMENTATION_H

namespace vfd
{
	struct RigidBodyDescription;

	struct RigidBodyData
	{
		RigidBodyData(const RigidBodyDescription& desc);

		glm::mat4x4 Transform;
	};
}

#endif // !RIGID_BODY_IMPLEMENTATION_H