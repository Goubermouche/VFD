#include "RadixSort.cuh"
#include "RadixSortKernel.cuh"
#include <utility>

namespace fe {
	extern "C" {
		void RadixSort(KeyValuePair* pairData0, KeyValuePair* pairData1, uint elements, uint bits)
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
				RadixSum << < NUM_BLOCKS, NUM_THREADS_PER_BLOCK, GRFSIZE >> > (pairData0, elements, elementsRounded, shift);
				RadixPrefixSum << <PREFIX_NUM_BLOCKS, PREFIX_NUM_THREADS_PER_BLOCK, PREFIX_GRFSIZE >> > ();
				RadixAddOffsetsAndShuffle << <NUM_BLOCKS, NUM_THREADS_PER_BLOCK, SHUFFLE_GRFSIZE >> > (pairData0, pairData1, elements, elementsRounded, shift);

				std::swap(pairData0, pairData1);
			}
		}
	}
}