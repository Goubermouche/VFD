#ifndef RADIX_SORT_KERNEL_CU_
#define RADIX_SORT_KERNEL_CU_

#include <stdio.h>
#include "RadixSort.cuh"

namespace fe {
	static const int NUM_SMS = 16;
	static const int NUM_THREADS_PER_SM = 192;
	static const int NUM_THREADS_PER_BLOCK = 64;
	//static const int NUM_THREADS = NUM_THREADS_PER_SM * NUM_SMS;
	static const int NUM_BLOCKS = (NUM_THREADS_PER_SM / NUM_THREADS_PER_BLOCK) * NUM_SMS;
	static const int RADIX = 8;	// Number of bits per radix sort pass
	static const int RADICES = 1 << RADIX; // Number of radices
	static const int RADIXMASK = RADICES - 1; // Mask for each radix sort pass
#if SIXTEEN
	static const int RADIXBITS = 16; // Number of bits to sort over
#else
	static const int RADIXBITS = 32; // Number of bits to sort over
#endif
	static const int RADIXTHREADS = 16;	// Number of threads sharing each radix counter
	static const int RADIXGROUPS = NUM_THREADS_PER_BLOCK / RADIXTHREADS; // Number of radix groups per CTA
	static const int TOTALRADIXGROUPS = NUM_BLOCKS * RADIXGROUPS; // Number of radix groups for each radix
	static const int SORTRADIXGROUPS = TOTALRADIXGROUPS * RADICES; // Total radix count
	static const int GRFELEMENTS = (NUM_THREADS_PER_BLOCK / RADIXTHREADS) * RADICES;
	static const int GRFSIZE = GRFELEMENTS * sizeof(uint);

	// Prefix sum variables
	static const int PREFIX_NUM_THREADS_PER_SM = NUM_THREADS_PER_SM;
	static const int PREFIX_NUM_THREADS_PER_BLOCK = PREFIX_NUM_THREADS_PER_SM;
	static const int PREFIX_NUM_BLOCKS = (PREFIX_NUM_THREADS_PER_SM / PREFIX_NUM_THREADS_PER_BLOCK) * NUM_SMS;
	static const int PREFIX_BLOCKSIZE = SORTRADIXGROUPS / PREFIX_NUM_BLOCKS;
	static const int PREFIX_GRFELEMENTS = PREFIX_BLOCKSIZE + 2 * PREFIX_NUM_THREADS_PER_BLOCK;
	static const int PREFIX_GRFSIZE = PREFIX_GRFELEMENTS * sizeof(uint);

	// Shuffle variables
	static const int SHUFFLE_GRFOFFSET = RADIXGROUPS * RADICES;
	static const int SHUFFLE_GRFELEMENTS = SHUFFLE_GRFOFFSET + PREFIX_NUM_BLOCKS;
	static const int SHUFFLE_GRFSIZE = SHUFFLE_GRFELEMENTS * sizeof(uint);

#define SDATA( index) CUT_BANK_CHECKER(sdata, index)

	// Prefix sum data
	extern uint gRadixSum[TOTALRADIXGROUPS * RADICES];
	__device__ uint dRadixSum[TOTALRADIXGROUPS * RADICES];
	extern uint gRadixBlockSum[PREFIX_NUM_BLOCKS];
	__device__ uint dRadixBlockSum[PREFIX_NUM_BLOCKS];

	extern __shared__ uint sRadixSum[];

	__global__ void RadixSum(KeyValuePair* pData, uint elements, uint elements_rounded_to_3072, uint shift);
	__global__ void RadixPrefixSum();
	__global__ void RadixAddOffsetsAndShuffle(KeyValuePair* pSrc, KeyValuePair* pDst, uint elements, uint elements_rounded_to_3072, int shift);
}

#endif // !RADIX_SORT_KERNEL_CU_
