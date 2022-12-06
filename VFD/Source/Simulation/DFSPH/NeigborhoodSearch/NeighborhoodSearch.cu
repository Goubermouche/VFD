#include "pch.h"
#include "NeighborhoodSearch.h"

#include "SearchDeviceData.h"
#include "PointSetImplementation.h"

namespace vfdcu {
	NeighborhoodSearch::NeighborhoodSearch(float searchRadius)
	{
		m_DeviceData = std::make_unique<SearchDeviceData>(searchRadius);
		SetRadius(searchRadius);
	}

	NeighborhoodSearch::~NeighborhoodSearch() {}

	unsigned int NeighborhoodSearch::AddPointSet(float const* x, std::size_t n, bool dynamic,
		bool searchNeighbors, bool findNeighbors, void* userData)
	{
		auto index = pointSets.size();
		pointSets.push_back(PointSet(x, n, dynamic, userData));
		m_ActivationTable.AddPointSet(searchNeighbors, findNeighbors);

		for (auto& pointSet : pointSets)
		{
			pointSet.m_Neighbors.resize(pointSets.size());
		}

		return static_cast<unsigned int>(index);
	}


	void NeighborhoodSearch::SetRadius(float r)
	{
		this->m_SearchRadius = r;
		m_DeviceData->setSearchRadius(r);
		m_Initialized = false;
	}

	void NeighborhoodSearch::ResizePointSet(unsigned int index, float const* x, std::size_t size)
	{
		pointSets[index].Resize(x, size);
	}

	void NeighborhoodSearch::UpdatePointSet(PointSet& pointSet)
	{
		pointSet.m_Implementation->CopyToDevice();
		m_DeviceData->ComputeMinMax(pointSet);
		m_DeviceData->ComputeCellInformation(pointSet);
	}

	void NeighborhoodSearch::FindNeighbors(bool points_changed_)
	{
		if (points_changed_ || !m_Initialized)
		{
			for (auto& pointSet : pointSets)
			{
				if (!m_Initialized || pointSet.IsDynamic())
				{
					UpdatePointSet(pointSet);
				}
			}
		}

		m_Initialized = true;

		for (unsigned int i = 0; i < pointSets.size(); i++)
		{
			for (unsigned int j = 0; j < pointSets.size(); j++)
			{
				if (m_ActivationTable.IsActive(i, j))
				{
					auto& queryPointSet = pointSets[i];
					auto& pointSet = pointSets[j];
					m_DeviceData->ComputeNeighborhood(queryPointSet, pointSet, j);
				}
			}
		}
	}

	void NeighborhoodSearch::UpdatePointSets()
	{
		for (unsigned int i = 0; i < pointSets.size(); i++)
		{
			UpdatePointSet(i);
		}
	}

	void NeighborhoodSearch::UpdatePointSet(int i)
	{
		UpdatePointSet(pointSets[i]);
	}
}