#ifndef NEIGHBORHOOD_SEARCH_H
#define NEIGHBORHOOD_SEARCH_H

// https://github.com/InteractiveComputerGraphics/cuNSearch

#include "ActivationTable.h"
#include "PointSet.h"

namespace vfdcu {
	class SearchDeviceData;

	class NeighborhoodSearch {
	private:
		std::vector<PointSet> pointSets;
	public:
		NeighborhoodSearch(float searchRadius);
		~NeighborhoodSearch();

		const PointSet& GetPointSet(unsigned int i) const
		{
			return pointSets[i];
		}

		PointSet& GetPointSet(unsigned int i)
		{
			return pointSets[i];
		}

		std::size_t  GetPointSetCount() const
		{
			return pointSets.size();
		}

		const std::vector<PointSet>& GetPointSets() const
		{
			return pointSets;
		}

		std::vector<PointSet>& GetPointSets()
		{
			return pointSets;
		}

		void ResizePointSet(unsigned int i, float const* x, std::size_t n);

		unsigned int AddPointSet(float const* x, std::size_t n, bool dynamic = true,
			bool searchNeighbors = true, bool findNeighbors = true, void* userData = nullptr);

		void FindNeighbors(bool pointsChanged = true);
		void UpdatePointSets();
		void UpdatePointSet(int i);
		void UpdateActivationTable();
		void ZSort();

		float GetRadius() const
		{
			return m_SearchRadius;
		}

		void SetRadius(float r);

		void SetActive(unsigned int i, unsigned int j, bool active)
		{
			m_ActivationTable.SetActive(i, j, active);
		}

		void SetActive(unsigned int i, bool searchNeighbor = true, bool findNeighbors = true)
		{
			m_ActivationTable.SetActive(i, searchNeighbor, findNeighbors);
		}

		void SetActive(bool active)
		{
			m_ActivationTable.SetActive(active);
		}

		bool IsActive(unsigned int i, unsigned int j) const
		{
			return m_ActivationTable.IsActive(i, j);
		}
	private:
		void UpdatePointSet(PointSet& pointSet);
	private:
		bool m_Initialized = false;
		float m_SearchRadius;
		ActivationTable m_ActivationTable;
		std::unique_ptr<SearchDeviceData> m_DeviceData;
	};
}

#endif // !NEIGHBORHOOD_SEARCH_H