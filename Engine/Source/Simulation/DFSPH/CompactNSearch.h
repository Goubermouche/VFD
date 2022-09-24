#ifndef COMPACT_N_SEARCH_H
#define COMPACT_N_SEARCH_H

// https://github.com/InteractiveComputerGraphics/CompactNSearch

#define INITIAL_NUMBER_OF_INDICES   50
#define INITIAL_NUMBER_OF_NEIGHBORS 50

namespace fe {
	struct HashKey {
		HashKey() = default;
		HashKey(int i, int j, int k)
		{
			this->k[0] = i, this->k[1] = j, this->k[2] = k;
		}

		HashKey& operator=(HashKey const& other)
		{
			k[0] = other.k[0];
			k[1] = other.k[1];
			k[2] = other.k[2];
			return *this;
		}

		bool operator==(HashKey const& other) const
		{
			return
				k[0] == other.k[0] &&
				k[1] == other.k[1] &&
				k[2] == other.k[2];
		}

		bool operator!=(HashKey const& other) const
		{
			return !(*this == other);
		}

		int k[3];
	};

	class Spinlock
	{
	public:

		void Lock()
		{
			while (m_Lock.test_and_set(std::memory_order_acquire));
		}

		void Unlock()
		{
			m_Lock.clear(std::memory_order_release);
		}

		Spinlock() = default;
		Spinlock(Spinlock const& other) {};
		Spinlock& operator=(Spinlock const& other) { return *this; }
	private:
		std::atomic_flag m_Lock = ATOMIC_FLAG_INIT;
	};

	class ActivationTable
	{
	private:
		std::vector<std::vector<unsigned char>> m_Table;
	public:
		bool operator==(ActivationTable const& other) const
		{
			return m_Table == other.m_Table;
		}

		bool operator!=(ActivationTable const& other) const
		{
			return !(m_Table == other.m_Table);
		}

		void AddPointSet(bool searchNeighbors = true, bool findNeighbors = true)
		{
			auto size = m_Table.size();
			for (auto i = 0u; i < size; i++)
			{
				m_Table[i].resize(size + 1);
				m_Table[i][size] = static_cast<unsigned char>(findNeighbors);
			}

			m_Table.resize(size + 1);
			m_Table[size].resize(size + 1);
			for (auto i = 0u; i < size + 1; i++) {
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
			for (auto i = 0u; i < size; i++) {
				for (auto j = 0u; j < size; j++) {
					m_Table[i][j] = static_cast<unsigned char>(active);
				}
			}
		}

		bool IsActive(unsigned int index1, unsigned int index2) const
		{
			return m_Table[index1][index2] != 0;
		}

		bool IsSearchingNeighbors(unsigned int const index) const
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

	struct SpatialHasher
	{
		std::size_t operator()(HashKey const& k) const
		{
			return static_cast<size_t>(
				static_cast<int64_t>(73856093) * static_cast<int64_t>(k.k[0]) ^
				static_cast<int64_t>(19349663) * static_cast<int64_t>(k.k[1]) ^
				static_cast<int64_t>(83492791) * static_cast<int64_t>(k.k[2]));
		}
	};

	class NeighborhoodSearch;
	class PointSet {
	public:
		PointSet(PointSet const& other) {
			*this = other;
		}

		PointSet& operator=(PointSet const& other)
		{
			m_X = other.m_X;
			m_N = other.m_N;
			m_Dynamic = other.m_Dynamic;
			m_UserData = other.m_UserData;

			m_Neighbors = other.m_Neighbors;
			m_Keys = other.m_Keys;
			m_OldKeys = other.m_OldKeys;

			m_SortTable = other.m_SortTable;

			m_Locks.resize(other.m_Locks.size());
			for (unsigned int i = 0; i < other.m_Locks.size(); ++i)
			{
				m_Locks[i].resize(other.m_Locks[i].size());
			}

			return *this;
		}

		float const* GetPoint(unsigned int i) const { return &m_X[3 * i]; }
		std::size_t GetPointCount() const { return m_N; }
		bool IsDynamic() const { return m_Dynamic; }
		std::size_t GetNeighborCount(unsigned int pointSet, unsigned int i) const;
		unsigned int GetNeighbor(unsigned int pointSet, unsigned int i, unsigned int k) const;
	private:
		friend NeighborhoodSearch;

		PointSet(float const* x, std::size_t n, bool dynamic, void* user_data = nullptr)
			: m_X(x), m_N(n), m_Dynamic(dynamic), m_UserData(user_data), m_Neighbors(n)
			, m_Keys(n, {
			std::numeric_limits<int>::lowest(),
			std::numeric_limits<int>::lowest(),
			std::numeric_limits<int>::lowest() })
		{
			m_OldKeys = m_Keys;
		}
	private:
		float const* m_X;
		std::size_t m_N;
		bool m_Dynamic;
		void* m_UserData;

		std::vector<std::vector<std::vector<unsigned int>>> m_Neighbors;
		std::vector<HashKey> m_Keys;
		std::vector<HashKey> m_OldKeys;
		std::vector < std::vector<Spinlock>> m_Locks;
		std::vector<unsigned int> m_SortTable;
	};

	struct PointID
	{
		unsigned int pointSetID;
		unsigned int pointID;

		bool operator==(PointID const& other) const
		{
			return pointID == other.pointID && pointSetID == other.pointSetID;
		}
	};

	struct HashEntry
	{
		HashEntry() : NSearchingPoints(0u)
		{
			indices.reserve(INITIAL_NUMBER_OF_INDICES);
		}

		HashEntry(PointID const& id) : NSearchingPoints(0u)
		{
			Add(id);
		}

		void Add(PointID const& id)
		{
			indices.push_back(id);
		}

		void Erase(PointID const& id)
		{
			auto it = std::find(indices.begin(), indices.end(), id);
			if (it != indices.end()) {
				indices.erase(it);
			}
		}

		unsigned int n_indices() const
		{
			return static_cast<unsigned int>(indices.size());
		}

		std::vector<PointID> indices;
		unsigned int NSearchingPoints;
	};

	class NeighborhoodSearch {
	public:
		NeighborhoodSearch(float r, bool eraseEmptyCells = false);
		virtual ~NeighborhoodSearch() = default;

		void FindNeighbors(bool pointsChanged = true);
		void UpdatePointSets();
		void UpdateActivationTable();

		void SetActive(bool active);
		void SetRadius(float value);

		PointSet& GetPointSet(unsigned int i) {
			return m_PointSets[i];
		}

		std::vector<PointSet>& GetPointSets() {
			return m_PointSets;
		}

		std::vector<PointSet> const& GetPointSets() const {
			return m_PointSets;
		}

		std::size_t GetPointSetCount() const {
			return m_PointSets.size();
		}

		unsigned int AddPointSet(float const* x, std::size_t n, bool isDynamic = true, bool searchNeighbors = true, bool findNeighbors = true, void* userData = nullptr);
	private:
		void Init();
		void UpdateHashTable(std::vector<unsigned int>& toDelete);
		void EraseEmptyEntries(std::vector<unsigned int>& const toDelete);
		void Query();

		HashKey GetCellIndex(float const* x) const;
	private:
		std::vector<PointSet> m_PointSets;
		ActivationTable m_ActivationTable;
		ActivationTable m_OldActivationTable;

		float m_InvCellSize;
		float m_R2;
		std::unordered_map<HashKey, unsigned int, SpatialHasher> m_Map;
		std::vector<HashEntry> m_Entries;

		bool m_EraseEmptyCells;
		bool m_Initialized;
	};
}

#endif // !COMPACT_N_SEARCH_H