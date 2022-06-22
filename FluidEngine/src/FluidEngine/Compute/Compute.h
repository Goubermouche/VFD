#ifndef COMPUTE_H_
#define COMPUTE_H_

#include "FluidEngine/Compute/ComputeAPI.h"

namespace fe {
	class Compute
	{
	public:
		static void Init();

		inline static ComputeAPIType GetAPI() {
			return ComputeAPI::GetAPIType();
		}

		static void SetAPI(ComputeAPIType api);

		static bool GetInitState() {
			ASSERT(s_ComputeAPI, "compute API not set!");
			return s_ComputeAPI->GetInitState();
		}

		static DeviceInfo GetDeviceInfo() {
			ASSERT(s_ComputeAPI, "compute API not set!");
			return s_ComputeAPI->GetDeviceInfo();
		}
		
	protected:
		static ComputeAPI* s_ComputeAPI;
	};
}

#endif // !COMPUTE_H_