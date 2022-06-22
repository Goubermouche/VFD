#ifndef COMPUTE_H_
#define COMPUTE_H_

#include "FluidEngine/Compute/ComputeAPI.h"

namespace fe {
	/// <summary>
	/// Wrapper class for all parallel processes.
	/// </summary>
	class Compute
	{
	public:
		/// <summary>
		/// Initializes the compute context based on the currently selected compute API.
		/// </summary>
		static void Init();

		inline static ComputeAPIType GetAPI() {
			return ComputeAPI::GetAPIType();
		}

		/// <summary>
		/// Sets the compute API, the Init() function should be called afterwards to update the context.
		/// </summary>
		/// <param name="api">New compute API.</param>
		static void SetAPI(ComputeAPIType api);

		/// <summary>
		/// Gets the compute init state.
		/// </summary>
		/// <returns>True if the compute context was created succesfully.</returns>
		static bool GetInitState() {
			ASSERT(s_ComputeAPI, "compute API not set!");
			return s_ComputeAPI->GetInitState();
		}

		/// <summary>
		/// Gets Device info for the current compute device.
		/// </summary>
		/// <returns>DeviceInfo struct containing information about the current compute device.</returns>
		static DeviceInfo GetDeviceInfo() {
			ASSERT(s_ComputeAPI, "compute API not set!");
			return s_ComputeAPI->GetDeviceInfo();
		}
		
	protected:
		static ComputeAPI* s_ComputeAPI;
	};
}

#endif // !COMPUTE_H_