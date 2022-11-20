#include "pch.h"
#include "CompactNSearch.h"

#include <ppl.h>

vfd::NeighborhoodSearch::NeighborhoodSearch(float r, bool eraseEmptyCells)
	: m_R2(r* r), m_InvCellSize(static_cast<float>(1.0 / r)), m_EraseEmptyCells(eraseEmptyCells), m_Initialized(false)
{
	ASSERT(r > 0.0f, "Neighborhood search may not be initialized with a zero or negative search radius.");
}

void vfd::NeighborhoodSearch::FindNeighbors(bool pointsChanged) {
	if (pointsChanged) {
		UpdatePointSets();
	}

	UpdateActivationTable();
	Query();
}

void vfd::NeighborhoodSearch::ZSort()
{
	for (PointSet& d : m_PointSets) {
		d.m_SortTable.resize(d.GetPointCount());
		std::iota(d.m_SortTable.begin(), d.m_SortTable.end(), 0);
		std::sort(d.m_SortTable.begin(), d.m_SortTable.end(), [&](unsigned int a, unsigned int b)
			{
				return GetZValue(GetCellIndex(d.GetPoint(a))) < GetZValue(GetCellIndex(d.GetPoint(b)));
			});
	}

	m_Initialized = false;
}


void vfd::NeighborhoodSearch::UpdatePointSets()
{
	if (!m_Initialized)
	{
		Init();
	}

	concurrency::parallel_for_each
	(m_PointSets.begin(), m_PointSets.end(), [&](PointSet& d)
		{
			if (d.IsDynamic())
			{
				d.m_Keys.swap(d.m_OldKeys);
				for (unsigned int i = 0; i < d.GetPointCount(); ++i)
				{
					d.m_Keys[i] = GetCellIndex(d.GetPoint(i));
				}
			}
		});

	std::vector<unsigned int> toDelete;
	if (m_EraseEmptyCells)
	{
		toDelete.reserve(m_Entries.size());
	}

	UpdateHashTable(toDelete);
	if (m_EraseEmptyCells)
	{
		EraseEmptyEntries(toDelete);
	}
}

void vfd::NeighborhoodSearch::UpdateActivationTable()
{
	if (m_ActivationTable != m_OldActivationTable)
	{
		for (auto& entry : m_Entries)
		{
			auto& n = entry.NSearchingPoints;
			n = 0u;
			for (auto const& idx : entry.indices)
			{
				if (m_ActivationTable.IsSearchingNeighbors(idx.pointSetID))
				{
					++n;
				}
			}
		}
		m_OldActivationTable = m_ActivationTable;
	}
}

void vfd::NeighborhoodSearch::SetActive(bool active)
{
	m_ActivationTable.SetActive(active);
	m_Initialized = false;
}

void vfd::NeighborhoodSearch::SetRadius(float value)
{
	m_R2 = value * value;
	m_InvCellSize = static_cast<float>(1.0 / value);
	m_Initialized = false;
}

unsigned int vfd::NeighborhoodSearch::AddPointSet(float const* x, std::size_t n, bool isDynamic, bool searchNeighbors, bool findNeighbors, void* userData)
{
	m_PointSets.push_back({ x, n, isDynamic, userData });
	m_ActivationTable.AddPointSet(searchNeighbors, findNeighbors);
	return static_cast<unsigned int>(m_PointSets.size() - 1);
}

void vfd::NeighborhoodSearch::ResizePointSet(unsigned int index, float const* x, std::size_t size)
{
	PointSet& point_set = m_PointSets[index];
	std::size_t old_size = point_set.GetPointCount();

	if (!m_Initialized)
	{
		ASSERT("point set not initialized!");
	}

	// Delete old entries. (Shrink)
	if (old_size > size)
	{
		std::vector<unsigned int> to_delete;
		if (m_EraseEmptyCells)
		{
			to_delete.reserve(m_Entries.size());
		}

		for (unsigned int i = static_cast<unsigned int>(size); i < old_size; i++)
		{
			HashKey const& key = point_set.m_Keys[i];
			auto it = m_Map.find(key);
			m_Entries[it->second].Erase({ index, i });
			if (m_ActivationTable.IsSearchingNeighbors(index)) {
				m_Entries[it->second].NSearchingPoints--;
			}

			if (m_EraseEmptyCells)
			{
				if (m_Entries[it->second].n_indices() == 0)
				{
					to_delete.push_back(it->second);
				}
			}
		}

		if (m_EraseEmptyCells)
		{
			EraseEmptyEntries(to_delete);
		}
	}

	point_set.Resize(x, size);

	// Insert new entries. (Grow)
	for (unsigned int i = static_cast<unsigned int>(old_size); i < point_set.GetPointCount(); i++)
	{
		HashKey key = GetCellIndex(point_set.GetPoint(i));
		point_set.m_Keys[i] = point_set.m_OldKeys[i] = key;
		auto it = m_Map.find(key);

		if (it == m_Map.end())
		{
			m_Entries.push_back({ { index, i } });
			if (m_ActivationTable.IsSearchingNeighbors(index)) {
				m_Entries.back().NSearchingPoints++;
			}
			m_Map[key] = static_cast<unsigned int>(m_Entries.size() - 1);
		}
		else
		{
			m_Entries[it->second].Add({ index, i });
			if (m_ActivationTable.IsSearchingNeighbors(index)) {
				m_Entries.back().NSearchingPoints++;
			}
		}
	}

	// Resize spinlock arrays.
	for (auto& l : point_set.m_Locks) {
		l.resize(point_set.GetPointCount());
	}
}

void vfd::NeighborhoodSearch::Init()
{
	m_Entries.clear();
	m_Map.clear();

	std::vector<HashKey> temp_keys;
	for (unsigned int j = 0; j < m_PointSets.size(); ++j)
	{
		PointSet& d = m_PointSets[j];
		d.m_Locks.resize(m_PointSets.size());
		for (auto& l : d.m_Locks) {
			l.resize(d.m_N);
		}

		for (unsigned int i = 0; i < d.m_N; i++)
		{
			HashKey const& key = GetCellIndex(d.GetPoint(i));
			d.m_Keys[i] = d.m_OldKeys[i] = key;

			auto it = m_Map.find(key);
			if (it == m_Map.end())
			{
				m_Entries.push_back({ { j, i } });
				if (m_ActivationTable.IsSearchingNeighbors(j)) {
					m_Entries.back().NSearchingPoints++;
				}
				temp_keys.push_back(key);
				m_Map[key] = static_cast<unsigned int>(m_Entries.size() - 1);
			}
			else
			{
				m_Entries[it->second].CreateAsset({ j, i });
				if (m_ActivationTable.IsSearchingNeighbors(j)) {
					m_Entries[it->second].NSearchingPoints++;
				}
			}
		}
	}

	m_Map.clear();
	for (unsigned int i = 0; i < m_Entries.size(); ++i)
	{
		m_Map.emplace(temp_keys[i], i);
	}

	m_Initialized = true;
}

void vfd::NeighborhoodSearch::UpdateHashTable(std::vector<unsigned int>& toDelete)
{
	for (unsigned int j = 0; j < m_PointSets.size(); ++j)
	{
		PointSet& d = m_PointSets[j];
		for (unsigned int i = 0; i < d.GetPointCount(); ++i)
		{
			if (d.m_Keys[i] == d.m_OldKeys[i]) continue;

			HashKey const& key = d.m_Keys[i];
			auto it = m_Map.find(key);
			if (it == m_Map.end())
			{
				m_Entries.push_back({ {j, i} });
				if (m_ActivationTable.IsSearchingNeighbors(j)) {
					m_Entries.back().NSearchingPoints++;
				}
				m_Map.insert({ key, static_cast<unsigned int>(m_Entries.size() - 1) });
			}
			else
			{
				HashEntry& entry = m_Entries[it->second];
				entry.CreateAsset({ j, i });
				if (m_ActivationTable.IsSearchingNeighbors(j)) {
					entry.NSearchingPoints++;
				}
			}

			unsigned int entry_index = m_Map[d.m_OldKeys[i]];
			m_Entries[entry_index].Erase({ j, i });
			if (m_ActivationTable.IsSearchingNeighbors(j)) {
				m_Entries[entry_index].NSearchingPoints--;
			}
			if (m_EraseEmptyCells)
			{
				if (m_Entries[entry_index].n_indices() == 0)
				{
					toDelete.push_back(entry_index);
				}
			}
		}
	}

	toDelete.erase(std::remove_if(toDelete.begin(), toDelete.end(),
		[&](unsigned int index)
		{
			return m_Entries[index].n_indices() != 0;
		}), toDelete.end());
	std::sort(toDelete.begin(), toDelete.end(), std::greater<unsigned int>());
}

void vfd::NeighborhoodSearch::EraseEmptyEntries(std::vector<unsigned int>& const toDelete)
{
	if (toDelete.empty()) {
		return;
	}

	m_Entries.erase(std::remove_if(m_Entries.begin(), m_Entries.end(), [](HashEntry const& entry)
		{
			return entry.indices.empty();
		}), m_Entries.end());

	{
		auto it = m_Map.begin();
		while (it != m_Map.end())
		{
			auto& kvp = *it;

			if (kvp.second <= toDelete.front() && kvp.second >= toDelete.back() &&
				std::binary_search(toDelete.rbegin(), toDelete.rend(), kvp.second))
			{
				it = m_Map.erase(it);
			}
			else
			{
				++it;
			}
		}
	}

	std::vector<std::pair<HashKey const, unsigned int>*> kvps(m_Map.size());
	std::transform(m_Map.begin(), m_Map.end(), kvps.begin(),
		[](std::pair<HashKey const, unsigned int>& kvp)
		{
			return &kvp;
		});

	concurrency::parallel_for_each
	(kvps.begin(), kvps.end(), [&](std::pair<HashKey const, unsigned int>* kvp_)
		{
			auto& kvp = *kvp_;

			for (unsigned int i = 0; i < toDelete.size(); ++i)
			{
				if (kvp.second >= toDelete[i])
				{
					kvp.second -= static_cast<unsigned int>(toDelete.size() - i);
					break;
				}
			}
		});
}

void vfd::NeighborhoodSearch::Query()
{
	for (unsigned int i = 0; i < m_PointSets.size(); i++)
	{
		PointSet& d = m_PointSets[i];
		d.m_Neighbors.resize(m_PointSets.size());
		for (unsigned int j = 0; j < d.m_Neighbors.size(); j++)
		{
			auto& n = d.m_Neighbors[j];
			n.resize(d.GetPointCount());
			for (auto& n_ : n)
			{
				n_.clear();
				if (m_ActivationTable.IsActive(i, j))
					n_.reserve(INITIAL_NUMBER_OF_NEIGHBORS);
			}
		}
	}

	std::vector<std::pair<HashKey const, unsigned int> const*> kvps(m_Map.size());
	std::transform(m_Map.begin(), m_Map.end(), kvps.begin(),
		[](std::pair<HashKey const, unsigned int> const& kvp)
		{
			return &kvp;
		});

	concurrency::parallel_for_each
	(kvps.begin(), kvps.end(), [&](std::pair<HashKey const, unsigned int> const* kvp_)
		{
			auto const& kvp = *kvp_;
			HashEntry const& entry = m_Entries[kvp.second];
			HashKey const& key = kvp.first;

			if (entry.NSearchingPoints == 0u)
			{
				return;
			}

			for (unsigned int a = 0; a < entry.n_indices(); ++a)
			{
				PointID const& va = entry.indices[a];
				PointSet& da = m_PointSets[va.pointSetID];
				for (unsigned int b = a + 1; b < entry.n_indices(); ++b)
				{
					PointID const& vb = entry.indices[b];
					PointSet& db = m_PointSets[vb.pointSetID];

					if (!m_ActivationTable.IsActive(va.pointSetID, vb.pointSetID) &&
						!m_ActivationTable.IsActive(vb.pointSetID, va.pointSetID))
					{
						continue;
					}

					float const* xa = da.GetPoint(va.pointID);
					float const* xb = db.GetPoint(vb.pointID);
					float tmp = xa[0] - xb[0];
					float l2 = tmp * tmp;
					tmp = xa[1] - xb[1];
					l2 += tmp * tmp;
					tmp = xa[2] - xb[2];
					l2 += tmp * tmp;

					if (l2 < m_R2)
					{
						if (m_ActivationTable.IsActive(va.pointSetID, vb.pointSetID))
						{
							da.m_Neighbors[vb.pointSetID][va.pointID].push_back(vb.pointID);
						}
						if (m_ActivationTable.IsActive(vb.pointSetID, va.pointSetID))
						{
							db.m_Neighbors[va.pointSetID][vb.pointID].push_back(va.pointID);
						}
					}
				}
			}
		});

	std::vector<std::array<bool, 27>> visited(m_Entries.size(), { false });
	std::vector<Spinlock> entry_locks(m_Entries.size());

	concurrency::parallel_for_each
	(kvps.begin(), kvps.end(), [&](std::pair<HashKey const, unsigned int> const* kvp_)
		{
			auto const& kvp = *kvp_;
			HashEntry const& entry = m_Entries[kvp.second];

			if (entry.NSearchingPoints == 0u) {
				return;
			}

			HashKey const& key = kvp.first;

			for (int dj = -1; dj <= 1; dj++) {
				for (int dk = -1; dk <= 1; dk++) {
					for (int dl = -1; dl <= 1; dl++) {
						int l_ind = 9 * (dj + 1) + 3 * (dk + 1) + (dl + 1);
						if (l_ind == 13)
						{
							continue;
						}
						entry_locks[kvp.second].Lock();
						if (visited[kvp.second][l_ind])
						{
							entry_locks[kvp.second].Unlock();
							continue;
						}
						entry_locks[kvp.second].Unlock();

						auto it = m_Map.find({ key.k[0] + dj, key.k[1] + dk, key.k[2] + dl });
						if (it == m_Map.end()) {
							continue;
						}

						std::array<unsigned int, 2> entry_ids{ {kvp.second, it->second} };
						if (entry_ids[0] > entry_ids[1]) {
							std::swap(entry_ids[0], entry_ids[1]);
						}

						entry_locks[entry_ids[0]].Lock();
						entry_locks[entry_ids[1]].Lock();

						if (visited[kvp.second][l_ind])
						{
							entry_locks[entry_ids[1]].Unlock();
							entry_locks[entry_ids[0]].Unlock();
							continue;
						}

						visited[kvp.second][l_ind] = true;
						visited[it->second][26 - l_ind] = true;

						entry_locks[entry_ids[1]].Unlock();
						entry_locks[entry_ids[0]].Unlock();

						for (unsigned int i = 0; i < entry.n_indices(); ++i)
						{
							PointID const& va = entry.indices[i];
							HashEntry const& entry_ = m_Entries[it->second];
							unsigned int n_ind = entry_.n_indices();
							for (unsigned int j = 0; j < n_ind; ++j)
							{
								PointID const& vb = entry_.indices[j];
								PointSet& db = m_PointSets[vb.pointSetID];

								PointSet& da = m_PointSets[va.pointSetID];

								if (!m_ActivationTable.IsActive(va.pointSetID, vb.pointSetID) &&
									!m_ActivationTable.IsActive(vb.pointSetID, va.pointSetID))
								{
									continue;
								}

								float const* xa = da.GetPoint(va.pointID);
								float const* xb = db.GetPoint(vb.pointID);
								float tmp = xa[0] - xb[0];
								float l2 = tmp * tmp;
								tmp = xa[1] - xb[1];
								l2 += tmp * tmp;
								tmp = xa[2] - xb[2];
								l2 += tmp * tmp;
								if (l2 < m_R2)
								{
									if (m_ActivationTable.IsActive(va.pointSetID, vb.pointSetID))
									{
										da.m_Locks[vb.pointSetID][va.pointID].Lock();
										da.m_Neighbors[vb.pointSetID][va.pointID].push_back(vb.pointID);
										da.m_Locks[vb.pointSetID][va.pointID].Unlock();
									}
									if (m_ActivationTable.IsActive(vb.pointSetID, va.pointSetID))
									{
										db.m_Locks[va.pointSetID][vb.pointID].Lock();
										db.m_Neighbors[va.pointSetID][vb.pointID].push_back(va.pointID);
										db.m_Locks[va.pointSetID][vb.pointID].Unlock();
									}
								}
							}
						}
					}
				}
			}
		});
}

vfd::HashKey vfd::NeighborhoodSearch::GetCellIndex(float const* x) const
{
	HashKey ret;
	for (unsigned int i = 0; i < 3; ++i)
	{
		if (x[i] >= 0.0) ret.k[i] = static_cast<int>(m_InvCellSize * x[i]);
		else ret.k[i] = static_cast<int>(m_InvCellSize * x[i]) - 1;
	}
	return ret;
}

std::size_t vfd::PointSet::GetNeighborCount(unsigned int pointSet, unsigned int i) const
{
	return static_cast<unsigned int>(m_Neighbors[pointSet][i].size());
}

unsigned int vfd::PointSet::GetNeighbor(unsigned int pointSet, unsigned int i, unsigned int k) const
{
	return m_Neighbors[pointSet][i][k];
}