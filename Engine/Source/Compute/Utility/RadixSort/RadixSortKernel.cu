#include "RadixSortKernel.cuh"

namespace fe {
	__global__ void RadixSum(KeyValuePair* pairData, unsigned int elements, unsigned int elementsRounded, unsigned int shift) {
		unsigned int pos = threadIdx.x;

		// Zero radix counts
		while (pos < s_GRFElementCount) {
			s_RadixSum[pos] = 0;
			pos += s_ThreadsPerBlock;
		}

		// Sum up data
		unsigned int tmod = threadIdx.x % s_RadixThreadCount;
		unsigned int tpos = threadIdx.x / s_RadixThreadCount;

		// Take the rounded element list size
		unsigned int element_fraction = elementsRounded / s_TotalRadixGroupCount;

		// Generate range 
		pos = (blockIdx.x * s_RadixGroupCount + tpos) * element_fraction;
		unsigned int end = pos + element_fraction;
		pos += tmod;

		__syncthreads();

		while (pos < end)
		{
			// Read first data element
			unsigned int key = 0;
			KeyValuePair kvp;
			if (pos < elements) {
				kvp = pairData[pos];
			}
			else {
				kvp.key = 0;
			}

			key = kvp.key;

			// Calculate position of radix counter
			unsigned int p = ((key >> shift) & s_RadixMask) * s_RadixGroupCount;

			// Increment radix counters
			unsigned int ppos = p + tpos;

			if (tmod == 1 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 2 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 3 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 4 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 5 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 0 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 6 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 7 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 8 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 9 && pos < elements) { s_RadixSum[ppos]++; }	__syncthreads();
			if (tmod == 10 && pos < elements) { s_RadixSum[ppos]++; } __syncthreads();
			if (tmod == 11 && pos < elements) { s_RadixSum[ppos]++; } __syncthreads();
			if (tmod == 12 && pos < elements) { s_RadixSum[ppos]++; } __syncthreads();
			if (tmod == 13 && pos < elements) { s_RadixSum[ppos]++; } __syncthreads();
			if (tmod == 14 && pos < elements) { s_RadixSum[ppos]++; } __syncthreads();
			if (tmod == 15 && pos < elements) { s_RadixSum[ppos]++; } __syncthreads();
			pos += s_RadixThreadCount;
		}

		__syncthreads();

		unsigned int offset = blockIdx.x * s_RadixGroupCount;
		unsigned int row = threadIdx.x / s_RadixGroupCount;
		unsigned int column = threadIdx.x % s_RadixGroupCount;
		while (row < s_Radices)
		{
			d_RadixSum[offset + row * s_TotalRadixGroupCount + column] = s_RadixSum[row * s_RadixGroupCount + column];
			row += s_ThreadsPerBlock / s_RadixGroupCount;
		}
	}
			
	__global__ void RadixPrefixSum()
	{
		unsigned int brow = blockIdx.x * (s_Radices / s_PrefixBlockCount);
		unsigned int drow = threadIdx.x / s_TotalRadixGroupCount;
		unsigned int dcolumn = threadIdx.x % s_TotalRadixGroupCount;
		unsigned int dpos = (brow + drow) * s_TotalRadixGroupCount + dcolumn;
		unsigned int end = ((blockIdx.x + 1) * (s_Radices / s_PrefixBlockCount)) * s_TotalRadixGroupCount;
		unsigned int srow = threadIdx.x / (s_PrefixBlockSize / s_PrefixThreadsPerBlock);
		unsigned int scolumn = threadIdx.x % (s_PrefixBlockSize / s_PrefixThreadsPerBlock);
		unsigned int spos = srow * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1) + scolumn;

		// Read (s_Radices / s_PrefixBlockCount) radix counts into the GRF alongside each other
		while (dpos < end) {
			s_RadixSum[spos] = d_RadixSum[dpos];
			spos += (s_PrefixThreadsPerBlock / (s_PrefixBlockSize / s_PrefixThreadsPerBlock)) * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1);
			dpos += (s_TotalRadixGroupCount / s_PrefixThreadsPerBlock) * s_TotalRadixGroupCount;
		}

		__syncthreads();

		// Perform preliminary sum on each thread's stretch of data
		int pos = threadIdx.x * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1);
		end = pos + (s_PrefixBlockSize / s_PrefixThreadsPerBlock);
		unsigned int sum = 0;

		while (pos < end) {
			sum += s_RadixSum[pos];
			s_RadixSum[pos] = sum;
			pos++;
		}

		__syncthreads();

		// Calculate internal offsets
		int m = (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1);
		pos = threadIdx.x * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1) + (s_PrefixBlockSize / s_PrefixThreadsPerBlock);
		s_RadixSum[pos] = s_RadixSum[pos - 1];
		__syncthreads();

		while (m < s_PrefixThreadsPerBlock * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1)) {
			int p = pos - m;
			unsigned int t = ((p > 0) ? s_RadixSum[p] : 0);
			__syncthreads();
			s_RadixSum[pos] += t;
			__syncthreads();
			m *= 2;
		}

		__syncthreads();

		// Add internal offsets to each thread's work data.
		pos = threadIdx.x * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1);
		end = pos + (s_PrefixBlockSize / s_PrefixThreadsPerBlock);
		int p = pos - 1;
		sum = ((p > 0) ? s_RadixSum[p] : 0);
		while (pos < end) {
			s_RadixSum[pos] += sum;
			pos++;
		}

		__syncthreads();

		// Write summed data back out to global memory in the same way as we read it in
		brow = blockIdx.x * (s_Radices / s_PrefixBlockCount);
		drow = threadIdx.x / s_TotalRadixGroupCount;
		dcolumn = threadIdx.x % s_TotalRadixGroupCount;
		srow = threadIdx.x / (s_PrefixBlockSize / s_PrefixThreadsPerBlock);
		scolumn = threadIdx.x % (s_PrefixBlockSize / s_PrefixThreadsPerBlock);
		dpos = (brow + drow) * s_TotalRadixGroupCount + dcolumn + 1;
		spos = srow * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1) + scolumn;
		end = ((blockIdx.x + 1) * s_Radices / s_PrefixBlockCount) * s_TotalRadixGroupCount;
		while (dpos < end) {
			d_RadixSum[dpos] = s_RadixSum[spos];
			dpos += (s_TotalRadixGroupCount / s_PrefixThreadsPerBlock) * s_TotalRadixGroupCount;
			spos += (s_PrefixThreadsPerBlock / (s_PrefixBlockSize / s_PrefixThreadsPerBlock)) * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1);
		}

		// Write last element to summation
		// Storing block sums in a separate array
		if (threadIdx.x == 0) {
			d_RadixBlockSum[blockIdx.x] = s_RadixSum[s_PrefixThreadsPerBlock * (s_PrefixBlockSize / s_PrefixThreadsPerBlock + 1) - 1];
			d_RadixSum[blockIdx.x * s_PrefixBlockSize] = 0;
		}
	}

	__global__ void RadixAddOffsetsAndShuffle(KeyValuePair* pairSource, KeyValuePair* pairDestination, unsigned int elements, unsigned int elementsRounded, int shift)
	{
		// Read offsets from previous blocks
		if (threadIdx.x == 0) {
			s_RadixSum[s_ShuffleGRFOffset] = 0;
		}

		if (threadIdx.x < s_PrefixBlockCount - 1) {
			s_RadixSum[s_ShuffleGRFOffset + threadIdx.x + 1] = d_RadixBlockSum[threadIdx.x];
		}

		__syncthreads();

		// Parallel prefix sum over block sums
		int pos = threadIdx.x;
		int n = 1;
		while (n < s_PrefixBlockCount)
		{
			int ppos = pos - n;
			unsigned int t0 = ((pos < s_PrefixBlockCount) && (ppos >= 0)) ? s_RadixSum[s_ShuffleGRFOffset + ppos] : 0;
			__syncthreads();
			if (pos < s_PrefixBlockCount)
				s_RadixSum[s_ShuffleGRFOffset + pos] += t0;
			__syncthreads();
			n *= 2;
		}

		// Read count data
		int row = threadIdx.x / s_RadixGroupCount;
		int column = threadIdx.x % s_RadixGroupCount;
		int spos = row * s_RadixGroupCount + column;
		int dpos = row * s_TotalRadixGroupCount + column + blockIdx.x * s_RadixGroupCount;

		while (spos < s_ShuffleGRFOffset) {
			s_RadixSum[spos] = d_RadixSum[dpos] + s_RadixSum[s_ShuffleGRFOffset + dpos / (s_TotalRadixGroupCount * s_Radices / s_PrefixBlockCount)];
			spos += s_ThreadsPerBlock;
			dpos += (s_ThreadsPerBlock / s_RadixGroupCount) * s_TotalRadixGroupCount;
		}

		__syncthreads();

		// Shuffle data
		unsigned int element_fraction = elementsRounded / s_TotalRadixGroupCount;
		int tmod = threadIdx.x % s_RadixThreadCount;
		int tpos = threadIdx.x / s_RadixThreadCount;

		pos = (blockIdx.x * s_RadixGroupCount + tpos) * element_fraction;
		unsigned int end = pos + element_fraction;
		pos += tmod;

		__syncthreads();

		while (pos < end) {
			KeyValuePair kvp;

			if (pos < elements)	{
				kvp = pairSource[pos];
			}
			else {
				kvp.key = 0;
			}

			unsigned int index;
			unsigned int p = ((kvp.key >> shift) & s_RadixMask) * s_RadixGroupCount;
			unsigned int ppos = p + tpos;

			if (tmod == 0 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 1 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 2 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 3 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 4 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 5 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 6 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 7 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 8 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 9 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 10 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 11 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 12 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 13 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 14 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
			if (tmod == 15 && pos < elements) { index = s_RadixSum[ppos]++; pairDestination[index] = kvp; } __syncthreads();
																										   
			pos += s_RadixThreadCount;
		}
		__syncthreads();
	}
}

