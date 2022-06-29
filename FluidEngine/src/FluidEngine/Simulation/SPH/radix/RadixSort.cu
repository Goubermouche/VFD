#include "RadixSort.cuh"
#include "RadixSortKernel.cuh"
#include <utility>

namespace fe {
	extern "C" {
		void RadixSort(KeyValuePair* pData0, KeyValuePair* pData1, uint elements, uint bits)
		{
			uint elementsRounded;
			int modVal = elements % 3072;
			
			if (modVal == 0) {
				elementsRounded = elements;
			}
			else {
				elementsRounded = elements + (3072 - modVal);
			}

			for (uint shift = 0; shift < bits; shift += RADIX)
			{
				// Generate per radix group sums radix counts across a radix group
				RadixSum << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, GRFSIZE >> > (pData0, elements, elementsRounded, shift);
				// Prefix sum in radix groups, and then between groups throughout a block
				RadixPrefixSum << <PREFIX_NUM_BLOCKS, PREFIX_NUM_THREADS_PER_BLOCK, PREFIX_GRFSIZE >> > ();
				// Sum the block offsets and then shuffle data into bins
				RadixAddOffsetsAndShuffle << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, SHUFFLE_GRFSIZE >> > (pData0, pData1, elements, elementsRounded, shift);

				KeyValuePair* pTemp = pData0;
				pData0 = pData1;
				pData1 = pTemp;
			}
		}
	}
}