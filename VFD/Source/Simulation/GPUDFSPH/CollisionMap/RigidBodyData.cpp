#include "pch.h"
#include "RigidBodyData.h"
#include "RigidBodyObject.h"

namespace vfd
{
	RigidBodyData::RigidBodyData(const RigidBodyDescription& desc)
		: Transform(desc.Transform)
	{
	}
}
