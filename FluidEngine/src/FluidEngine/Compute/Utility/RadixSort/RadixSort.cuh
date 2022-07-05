#ifndef RADIX_SORT_CUH_
#define RADIX_SORT_CUH_

#include <host_defines.h>

// Use 16 bit keys/values
#define SIXTEEN 0

#if SIXTEEN
typedef struct __align__(4) {
	ushort key;
	ushort value;
#else
typedef struct __align__(8) {
	unsigned int key;
	unsigned int value;
#endif
} KeyValuePair;

namespace fe {
	extern "C" {
		/// <summary>
		/// A basic radix sort function inspired by the CUDA playground implementation. 
		/// </summary>
		/// <param name="pData0"></param>
		/// <param name="pData1"></param>
		/// <param name="elements"></param>
		/// <param name="bits"></param>RadixSort
		void RadixSort(KeyValuePair* pairData0, KeyValuePair* pairData1, unsigned int elements, unsigned int bits);
	}
}

#endif // !RADIX_SORT_CUH_