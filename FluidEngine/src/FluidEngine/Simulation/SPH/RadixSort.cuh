#ifndef RADIX_SORT_CUH_
#define RADIX_SORT_CUH_

#include <host_defines.h>

// Use 16 bit keys/values
#define SIXTEEN 0

typedef unsigned int uint;
typedef unsigned short ushort;

#if SIXTEEN
typedef struct __align__(4) {
	ushort key;
	ushort value;
#else
typedef struct __align__(8) {
	uint key;
	uint value;
#endif
} KeyValuePair;

namespace fe {
	extern "C" {
		void RadixSort(KeyValuePair* pairData0, KeyValuePair* pairData1, uint elements, uint bits);
	}
}

#endif // !RADIX_SORT_CUH_