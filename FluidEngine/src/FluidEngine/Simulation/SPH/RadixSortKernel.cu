#include "RadixSortKernel.cuh"

namespace fe {

	__global__ void RadixSum(KeyValuePair* pairData, uint elements, uint elementsRounded, uint shift) {
		uint position = threadIdx.x;

		while (position < GRFELEMENTS)
		{
			sRadixSum[position] = 0;
			position += NUM_THREADS_PER_BLOCK;
		}

		uint tMod = threadIdx.x % RADIXTHREADS;
		uint tPosition = threadIdx.x / RADIXTHREADS;
		uint elementFraction = elementsRounded / TOTALRADIXGROUPS;
		position = (blockIdx.x * RADIXGROUPS + tPosition) * elementFraction;
		uint end = position + elementFraction;
		position += tMod;

		__syncthreads();

		while (position < end) {
			uint key = 0;
			KeyValuePair kvp;
			if (position < elements) {
				kvp = pairData[position];
			}
			else {
				kvp.key = 0;
			}
			key = kvp.key;

			uint p = ((key >> shift) & RADIXMASK) * RADIXGROUPS;
			uint ppos = p + tPosition;

			if (tMod == 0 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 1 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 2 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 3 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 4 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 5 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 6 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 7 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 8 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 9 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 10 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 11 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 12 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 13 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 14 && position < elements) { sRadixSum[ppos]++; } __syncthreads();
			if (tMod == 15 && position < elements) { sRadixSum[ppos]++; } __syncthreads();

			position += RADIXTHREADS;
		}

		__syncthreads();
		__syncthreads(); // CHECK

		uint offset = blockIdx.x * RADIXGROUPS;
		uint row = blockIdx.x / RADIXGROUPS;
		uint column = blockIdx.x % RADIXGROUPS;
		
		while (row < RADICES) {
			dRadixSum[offset + row * TOTALRADIXGROUPS + column] = sRadixSum[row * RADIXGROUPS + column];
			row += NUM_THREADS_PER_BLOCK / RADIXGROUPS;
		}
	}

	__global__ void RadixPrefixSum()
	{
		uint bRow = blockIdx.x * (RADICES / PREFIX_NUM_BLOCKS);
		uint dRow = threadIdx.x / TOTALRADIXGROUPS;
		uint dColumn = threadIdx.x % TOTALRADIXGROUPS;
		uint dPosition = (bRow + dRow) * TOTALRADIXGROUPS + dColumn;
		uint end = ((blockIdx.x + 1) * (RADICES / PREFIX_NUM_BLOCKS)) * TOTALRADIXGROUPS;
		uint sRow = threadIdx.x / (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK);
		uint sColumn = threadIdx.x % (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK);
		uint sPosition = sRow * (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1) + sColumn;

		while (dPosition < end)
		{
			sRadixSum[sPosition] = dRadixSum[dPosition];
			sPosition += (PREFIX_NUM_THREADS_PER_BLOCK / (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK)) *
				(PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1);
			dPosition += (TOTALRADIXGROUPS / PREFIX_NUM_THREADS_PER_BLOCK) * TOTALRADIXGROUPS;
		}

		__syncthreads();

		int position = threadIdx.x * (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1);
		end = position + (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK);
		uint sum = 0;

		while (position < end) {
			sum += sRadixSum[position];
			sRadixSum[position] = sum;
			position++;
		}

		__syncthreads();

		int m = (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1);
		position = threadIdx.x * (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1) +
			(PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK);
		sRadixSum[position] = sRadixSum[position - 1];

		__syncthreads();

		while (m < PREFIX_NUM_THREADS_PER_BLOCK * (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1)) {
			int p = position - m;
			uint t = ((p > 0) ? sRadixSum[p] : 0);
			__syncthreads();
			sRadixSum[position] += t;
			m *= 2;
		}

		__syncthreads();

		position = threadIdx.x * (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1);
		end = position + (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK);
		int p = position - 1;
		sum = ((p > 0) ? sRadixSum[p] : 0);

		while (position < end) {
			sRadixSum[position] += sum;
			position++;
		}

		__syncthreads();

		bRow = blockIdx.x * (RADICES / PREFIX_NUM_BLOCKS);
		dRow = threadIdx.x / TOTALRADIXGROUPS;
		dColumn = threadIdx.x % TOTALRADIXGROUPS;
		sRow = threadIdx.x / (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK);
		sColumn = threadIdx.x % (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK);
		dPosition = (bRow + dRow) * TOTALRADIXGROUPS + dColumn + 1;
		sPosition = sRow * (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1) + sColumn;
		end = ((blockIdx.x + 1) * RADICES / PREFIX_NUM_BLOCKS) * TOTALRADIXGROUPS;

		while (dPosition < end)
		{
			dRadixBlockSum[dPosition] = sRadixSum[sPosition];
			dPosition += (TOTALRADIXGROUPS / PREFIX_NUM_THREADS_PER_BLOCK) * TOTALRADIXGROUPS;
			sPosition += (PREFIX_NUM_THREADS_PER_BLOCK / (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK)) *
				(PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1);
		}

		if (threadIdx.x == 0) {
			dRadixBlockSum[blockIdx.x] = sRadixSum[PREFIX_NUM_THREADS_PER_BLOCK * (PREFIX_BLOCKSIZE / PREFIX_NUM_THREADS_PER_BLOCK + 1) - 1];
			dRadixSum[blockIdx.x * PREFIX_BLOCKSIZE] = 0;
		}
	}

	__global__ void RadixAddOffsetsAndShuffle(KeyValuePair* pairSrc, KeyValuePair* pairDst, uint elements, uint elementsRounded, int shift)
	{
		if (threadIdx.x == 0) {
			sRadixSum[SHUFFLE_GRFOFFSET] = 0;
		}

		if (threadIdx.x < PREFIX_NUM_BLOCKS - 1) {
			sRadixSum[SHUFFLE_GRFOFFSET + threadIdx.x + 1] = dRadixBlockSum[threadIdx.x];
		}

		__syncthreads();

		int position = threadIdx.x;
		int n = 1;

		while (n < PREFIX_NUM_BLOCKS) {
			int pPosition = position - 1;
			uint t0 = ((position < PREFIX_NUM_BLOCKS) && (pPosition >= 0)) ? sRadixSum[SHUFFLE_GRFOFFSET + pPosition] : 0;
			__syncthreads();
			if (position < PREFIX_NUM_BLOCKS)
				sRadixSum[SHUFFLE_GRFOFFSET + position] += t0;
			__syncthreads();
			n *= 2;
		}

		int row = threadIdx.x / RADIXGROUPS;
		int column = threadIdx.x % RADIXGROUPS;
		int sPosition = row * RADIXGROUPS + column;
		int dPosition = row * TOTALRADIXGROUPS + column + blockIdx.x * RADIXGROUPS;

		while (sPosition < SHUFFLE_GRFOFFSET)
		{
			sRadixSum[sPosition] = dRadixSum[dPosition] + sRadixSum[SHUFFLE_GRFOFFSET + dPosition / (TOTALRADIXGROUPS * RADICES / PREFIX_NUM_BLOCKS)];
			sPosition += NUM_THREADS_PER_BLOCK;
			dPosition += (NUM_THREADS_PER_BLOCK / RADIXGROUPS) * TOTALRADIXGROUPS;
		}

		__syncthreads();

		uint elementFraction = elementsRounded / TOTALRADIXGROUPS;
		int tMod = threadIdx.x % RADIXTHREADS;
		int tPosition = threadIdx.x / RADIXTHREADS;
		position = (blockIdx.x * RADIXGROUPS + tPosition) * elementFraction;
		uint end = position + elementFraction;
		position += tMod;

		__syncthreads();

		while (position < end) {
			KeyValuePair kvp;
#if 1 
			if (position < elements)	{
				kvp = pairSrc[position];
			}
			else {
				kvp.key = 0;
			}

#else
			int2 kvpf2;

			if (position < elements)
			{
				kvpf2 = ((int2*)pairSrc)[position];
			}
			else {
				kvpf2.x = 0;
			}

			kvp.key = kvpf2.x;
			kvp.value = kvpf2.y;
#endif  
			uint index;
			uint p = ((kvp.key >> shift) & RADIXMASK) * RADIXGROUPS;

			uint ppos = p + tPosition;
			if (tMod == 0 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 1 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 2 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 3 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 4 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 5 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 6 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 7 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 8 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 9 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 10 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 11 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 12 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 13 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 14 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();
			if (tMod == 15 && position < elements) { index = sRadixSum[ppos]++;	pairDst[index] = kvp; }	__syncthreads();

			position += RADIXTHREADS;
		}

		__syncthreads();
	}
}

