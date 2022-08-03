#ifndef RADIX_SORT_KERNEL_CU_
#define RADIX_SORT_KERNEL_CU_

#include <stdio.h>
#include "RadixSort.cuh"

namespace fe {
	static const int s_SMCount = 16;
	static const int s_ThreadsPerSM = 192;
	static const int s_ThreadsPerBlock = 64;
	static const int s_BlockCount = (s_ThreadsPerSM / s_ThreadsPerBlock) * s_SMCount;
	static const int s_RadixCount = 8;
	static const int s_Radices = 1 << s_RadixCount;
	static const int s_RadixMask = s_Radices - 1;
	static const int s_RadixBitCount = 32;
	static const int s_RadixThreadCount = 16;
	static const int s_RadixGroupCount = s_ThreadsPerBlock / s_RadixThreadCount;
	static const int s_TotalRadixGroupCount = s_BlockCount * s_RadixGroupCount;
	static const int s_SortRadixGroupCount = s_TotalRadixGroupCount * s_Radices;
	static const int s_GRFElementCount = (s_ThreadsPerBlock / s_RadixThreadCount) * s_Radices;
	static const int s_GRFSize = s_GRFElementCount * sizeof(unsigned int);

	// Prefix sum variables
	static const int s_PrefixThreadsPerSM = s_ThreadsPerSM;
	static const int s_PrefixThreadsPerBlock = s_PrefixThreadsPerSM;
	static const int s_PrefixBlockCount = (s_PrefixThreadsPerSM / s_PrefixThreadsPerBlock) * s_SMCount;
	static const int s_PrefixBlockSize = s_SortRadixGroupCount / s_PrefixBlockCount;
	static const int s_PrefixGRFElementCount = s_PrefixBlockSize + 2 * s_PrefixThreadsPerBlock;
	static const int s_PrefixGRFSize = s_PrefixGRFElementCount * sizeof(unsigned int);

	// Shuffle variables
	static const int s_ShuffleGRFOffset = s_RadixGroupCount * s_Radices;
	static const int s_ShuffleGRFElementCount = s_ShuffleGRFOffset + s_PrefixBlockCount;
	static const int s_ShuffleGRFSize = s_ShuffleGRFElementCount * sizeof(unsigned int);

#define SDATA( index) CUT_BANK_CHECKER(sdata, index)

	// Prefix sum data
	__device__ unsigned int d_RadixSum[s_TotalRadixGroupCount * s_Radices];
	__device__ unsigned int d_RadixBlockSum[s_PrefixBlockCount];

	extern __shared__ unsigned int s_RadixSum[];

	__global__ void RadixSum(KeyValuePair* pairData, unsigned int elements, unsigned int elementsRounded, unsigned int shift);
	__global__ void RadixPrefixSum();
	__global__ void RadixAddOffsetsAndShuffle(KeyValuePair* pairSource, KeyValuePair* pairDestination, unsigned int elements, unsigned int elementsRounded, int shift);
}

#endif // !RADIX_SORT_KERNEL_CU_