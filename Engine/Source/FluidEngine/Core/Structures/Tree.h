#ifndef TREE_H_
#define TREE_H_

#include "FluidEngine/Core/Structures/BoundingBox.h"

namespace fe {
	template <typename T>
	class Tree
	{
	public:
		struct Node
		{
			Node(uint32_t b_, uint32_t n_)
				: children({{-1, -1} }), begin(b_), n(n_) 
			{}
			Node() = default;

			bool IsLeaf() const {
				return children[0] < 0 && children[1] < 0;
			}
			
			std::array<int, 2> children;
			uint32_t begin;
			uint32_t n;
		};

		struct QueueItem {
			uint32_t n, d;
		};

		using TraversalPredicate = std::function<bool(uint32_t nodeIndex, uint32_t depth)>;
		using TraversalCallback = std::function <void(uint32_t nodeIndex, uint32_t depth)>;
		using TraversalPriorityLess = std::function<bool(std::array<int, 2> const& nodes)>;
		using TraversalQueue = std::queue<QueueItem>;

		Tree(std::size_t n)
			: m_List(n) {}

		virtual ~Tree() {}

		Node const& GetNode(uint32_t i) const {
			return m_Nodes[i];
		}

		T const& GetType(uint32_t i) const { 
			return m_Types[i];
		}
		uint32_t GetEntity(uint32_t i) const { 
			return m_List[i];
		}

		void Construct() {
			m_Nodes.clear();
			m_Types.clear();
			if (m_List.empty()) {
				return;
			}

			std::iota(m_List.begin(), m_List.end(), 0);

			BoundingBox box = BoundingBox{};
			for (uint32_t i = 0u; i < m_List.size(); ++i) {
				box.Extend(GetEntityPosition(i));
			}

			auto ni = AddNode(0, static_cast<uint32_t>(m_List.size()));
			Construct(ni, box, 0, static_cast<uint32_t>(m_List.size()));
		}

		void Update() {
			TraverseDepthFirst(
				[&](uint32_t, uint32_t) { return true; },
				[&](uint32_t nodeIndex, uint32_t)
				{
					auto const& nd = GetNode(nodeIndex);
					Calculate(nd.begin, nd.n, GetType(nodeIndex));
				}
			);
		}

		void TraverseDepthFirst(TraversalPredicate predicate, TraversalCallback callback, TraversalPriorityLess const& priority = nullptr) const {
			if (m_Nodes.empty()) {
				return;
			}

			if (predicate(0, 0)) {
				TraverseDepthFirst(0, 0, predicate, callback, priority);
			}
		}

		void TraverseBreadthFirst(TraversalPredicate const& predicate, TraversalCallback const& callback, uint32_t startNode = 0, TraversalPriorityLess const& priority = nullptr, TraversalQueue& pending = TraversalQueue()) const {
			callback(startNode, 0);

			if (predicate(startNode, 0)) {
				pending.push({ startNode, 0 });
			}

			TraverseBreadthFirst(pending, predicate, callback, priority);
		}
	protected:
		void Construct(uint32_t node, BoundingBox const& box, uint32_t b, uint32_t n) {
			if (n < 10) {
				return;
			}

			int maxDir = 0;
			glm::vec3 d = box.Diagonal();
			if (d.y >= d.x && d.y >= d.z) {
				maxDir = 1;
			}
			else if (d.z >= d.x && d.z >= d.y) {
				maxDir = 2;
			}

			std::sort(m_List.begin() + b, m_List.begin() + b + n,
				[&](uint32_t a, uint32_t b)
				{
					return GetEntityPosition(a)[maxDir] < GetEntityPosition(b)[maxDir];
				}
			);

			uint32_t hal = n / 2;
			uint32_t n0 = AddNode(b, hal);
			uint32_t n1 = AddNode(b + hal, n - hal);
			m_Nodes[node].children[0] = n0;
			m_Nodes[node].children[1] = n1;

			float c = 0.5f * (GetEntityPosition(m_List[b + hal - 1])[maxDir] +	GetEntityPosition(m_List[b + hal])[maxDir]);
			BoundingBox leftBox = box;
			leftBox.max[maxDir] = c;
			BoundingBox rightBox = box; 
			rightBox.min[maxDir] = c;

			Construct(m_Nodes[node].children[0], leftBox, b, hal);
			Construct(m_Nodes[node].children[1], rightBox, b + hal, n - hal);
		}

		void TraverseDepthFirst(uint32_t nodeIndex, uint32_t depth, TraversalPredicate predicate, TraversalCallback callback, TraversalPriorityLess const& priority) const {
			Node const& node = m_Nodes[nodeIndex];

			callback(nodeIndex, depth);
			auto isPredicate = predicate(nodeIndex, depth);
			if (!node.IsLeaf() && isPredicate)
			{
				if (priority && !priority(node.children))
				{
					TraverseDepthFirst(m_Nodes[nodeIndex].children[1], depth + 1, predicate, callback, priority);
					TraverseDepthFirst(m_Nodes[nodeIndex].children[0], depth + 1, predicate, callback, priority);
				}
				else
				{
					TraverseDepthFirst(m_Nodes[nodeIndex].children[0], depth + 1, predicate, callback, priority);
					TraverseDepthFirst(m_Nodes[nodeIndex].children[1], depth + 1, predicate, callback, priority);
				}
			}
		}

		void TraverseBreadthFirst(TraversalQueue& pending, TraversalPredicate const& predicate, TraversalCallback const& callback, TraversalPriorityLess const& priority = nullptr) const {
			while (!pending.empty())
			{
				T n = pending.front().n;
				T d = pending.front().d;
				T const& node = m_Nodes[n];
				pending.pop();

				callback(n, d);
				bool isPredicate = predicate(n, d);
				if (!node.IsLeaf() && isPredicate)
				{
					if (priority && !priority(node.children))
					{
						pending.push({ static_cast<uint32_t>(node.children[1]), d + 1 });
						pending.push({ static_cast<uint32_t>(node.children[0]), d + 1 });
					}
					else
					{
						pending.push({ static_cast<uint32_t>(node.children[0]), d + 1 });
						pending.push({ static_cast<uint32_t>(node.children[1]), d + 1 });
					}
				}
			}
		}

		uint32_t AddNode(uint32_t b, uint32_t n) {
			T type;
			Calculate(b, n, type);
			m_Types.push_back(type);
			m_Nodes.push_back({ b, n });
			return static_cast<uint32_t>(m_Nodes.size() - 1);
		}

		virtual glm::vec3 const& GetEntityPosition(uint32_t i) const = 0;
		virtual void Calculate(uint32_t b, uint32_t n, T& type) const = 0;
	protected:
		std::vector<uint32_t> m_List;
		std::vector<Node> m_Nodes;
		std::vector<T> m_Types;
	};
}

#endif // !TREE_H_
