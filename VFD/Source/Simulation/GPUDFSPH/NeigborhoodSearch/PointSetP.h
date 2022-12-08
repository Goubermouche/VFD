#ifndef POINT_SET_P_H
#define POINT_SET_P_H

#include "pch.h"
#include "../DFSPHParticle.h"

namespace vfd {
	class NeighborhoodSearch;
	class PointSetImplementation;
	class SearchDeviceData;

	class PointSet {
		struct NeighborSet {
			NeighborSet()
			{
				NeighborCountAllocationSize = 0u;
				ParticleCountAllocationSize = 0u;
				Counts = nullptr;
				Offsets = nullptr;
				Neighbors = nullptr;
			}

			unsigned int NeighborCountAllocationSize;
			unsigned int ParticleCountAllocationSize;
			unsigned int* Counts;
			unsigned int* Offsets;
			unsigned int* Neighbors;
		};
	public:
		PointSet(const PointSet& other);

		inline std::size_t GetNeighborCount(unsigned int pointSet, unsigned int i) const
		{
			return m_Neighbors[pointSet].Counts[i];
		}

		inline unsigned int GetNeighbor(unsigned int pointSet, unsigned int i, unsigned int k) const
		{
			//Return index of the k-th neighbor to point i (of the given point set)
			const auto& neighborSet = m_Neighbors[pointSet];
			return neighborSet.Neighbors[neighborSet.Offsets[i] + k];
		}

		inline unsigned int* GetNeighborList(unsigned int pointSet, unsigned int i) const
		{
			//Return index of the k-th neighbor to point i (of the given point set)
			const auto& neighborSet = m_Neighbors[pointSet];
			return &neighborSet.Neighbors[neighborSet.Offsets[i]];
		}

		std::size_t GetPointCount() const
		{
			return m_PointCount;
		}

		bool IsDynamic() const
		{
			return m_Dynamic;
		}

		void SetDynamic(bool v)
		{
			m_Dynamic = v;
		}

		DFSPHParticle const* GetPoints()
		{
			return m_Points;
		}

		void* GetUserData()
		{
			return m_UserData;
		}

		void SortField(DFSPHParticle* particles);
	private:
		friend NeighborhoodSearch;
		friend SearchDeviceData;

	    std::unique_ptr<PointSetImplementation> m_Implementation;

		PointSet(const DFSPHParticle* x, std::size_t n, bool dynamic, void* userData = nullptr);

		void Resize(DFSPHParticle const* x, std::size_t n);

		const DFSPHParticle* GetPoint(unsigned int i) const {
			return &m_Points[i]; 
		}

		const DFSPHParticle* m_Points;
		std::size_t m_PointCount;
		bool m_Dynamic;
		void* m_UserData;

		std::vector<unsigned int> m_SortedIndices;
		std::vector<NeighborSet> m_Neighbors;
	};

	//template<typename T>
	//inline void PointSet::SortField(T* lst) const
	//{
	//	std::vector<T> tmp(lst, lst + m_SortedIndices.size());

	//	std::transform(m_SortedIndices.begin(), m_SortedIndices.end(), lst, [&](int i)
	//	{ 
	//		return tmp[i];
	//	});
	//}
}

#endif // !POINT_SET_H