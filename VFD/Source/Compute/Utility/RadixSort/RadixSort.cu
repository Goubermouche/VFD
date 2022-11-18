#include "RadixSort.cuh"
#include "RadixSortKernel.cuh"
#include <utility>

namespace vfd {
	extern "C" {
		void RadixSort(KeyValuePair* pairData0, KeyValuePair* pairData1, unsigned int elements, unsigned int bits)
		{
			unsigned int elementsRounded;
			int modVal = elements % 3072;
			
			if (modVal == 0) {
				elementsRounded = elements;
			}
			else {
				elementsRounded = elements + (3072 - modVal);
			}

			for (unsigned int shift = 0; shift < bits; shift += s_RadixCount)
			{
				RadixSum <<< s_BlockCount, s_ThreadsPerBlock, s_GRFSize >>> (pairData0, elements, elementsRounded, shift);
				RadixPrefixSum <<< s_PrefixBlockCount, s_PrefixThreadsPerBlock, s_PrefixGRFSize >>> ();
				RadixAddOffsetsAndShuffle <<< s_BlockCount, s_ThreadsPerBlock, s_ShuffleGRFSize >>> (pairData0, pairData1, elements, elementsRounded, shift);

				std::swap(pairData0, pairData1);
			}
		}
	}
}