#ifndef NEIGHBOR_SET_H
#define NEIGHBOR_SET_H

#include "pch.h"

namespace vfd
{
	struct NeighborSet {
		__device__ unsigned int GetNeighborCount(unsigned int i) const
		{
			return Counts[i];
		}

		__device__ unsigned int GetNeighbor(unsigned int i, unsigned int j) const
		{
			return Neighbors[Offsets[i] + j];
		}

		__device__ unsigned int* GetNeighborList(unsigned int i) const
		{
			return &Neighbors[Offsets[i]];
		}

	private:
		friend class ParticleSearch;

		unsigned int* Counts = nullptr;
		unsigned int* Offsets = nullptr;
		unsigned int* Neighbors = nullptr;
	};
}

#endif // !NEIGHBOR_SET_H