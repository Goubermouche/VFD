#ifndef ACTIVATION_TABLE_H
#define ACTIVATION_TABLE_H

#include "pch.h"

namespace vfdcu {
	class ActivationTable {
	private:
		std::vector<std::vector<unsigned char>> m_Table;
	public:
		bool operator==(const ActivationTable& other) const
		{
			return m_Table == other.m_Table;
		}

		bool operator!=(ActivationTable const& other) const
		{
			return !(m_Table == other.m_Table);
		}

		void AddPointSet(bool searchNeighbors = true, bool findNeighbors = true)
		{
			// add column to each row
			auto size = m_Table.size();
			for (auto i = 0u; i < size; i++)
			{
				m_Table[i].resize(size + 1);
				m_Table[i][size] = static_cast<unsigned char>(findNeighbors);
			}

			// add new row
			m_Table.resize(size + 1);
			m_Table[size].resize(size + 1);
			for (auto i = 0u; i < size + 1; i++)
			{
				m_Table[size][i] = static_cast<unsigned char>(searchNeighbors);
			}
		}

		void SetActive(unsigned int index1, unsigned int index2, bool active)
		{
			m_Table[index1][index2] = static_cast<unsigned char>(active);
		}

		void SetActive(unsigned int index, bool searchNeighbors = true, bool findNeighbors = true)
		{
			auto size = m_Table.size();

			for (auto i = 0u; i < size; i++)
			{
				m_Table[i][index] = static_cast<unsigned char>(findNeighbors);
				m_Table[index][i] = static_cast<unsigned char>(searchNeighbors);
			}

			m_Table[index][index] = static_cast<unsigned char>(searchNeighbors && findNeighbors);
		}

		void SetActive(bool active)
		{
			auto size = m_Table.size();
			for (auto i = 0u; i < size; i++)
			{
				for (auto j = 0u; j < size; j++)
				{
					m_Table[i][j] = static_cast<unsigned char>(active);
				}
			}
		}

		bool IsActive(unsigned int index1, unsigned int index2) const
		{
			return m_Table[index1][index2] != 0;
		}

		bool IsSearchingNeighbors(const unsigned int index) const
		{
			for (auto i = 0u; i < m_Table[index].size(); i++)
			{
				if (m_Table[index][i])
				{
					return true;
				}
			}

			return false;
		}
	};
}

#endif // !ACTIVATION_TABLE_H
