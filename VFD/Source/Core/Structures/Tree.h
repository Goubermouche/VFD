#ifndef TREE_H
#define TREE_H

#include "Core/Structures/BoundingBox.h"

namespace vfd {
	/// <summary>
	/// Basic tree x node hierarchy of type T. 
	/// </summary>
	/// <typeparam name="T"></typeparam>
	template <typename T>
	class Tree
	{
	public:
		struct Node
		{
			Node(const unsigned int b, const unsigned int n)
				: children({{-1, -1} }), begin(b), n(n) 
			{}
			Node() = default;
			~Node() = default;

			[[nodiscard]]
			bool IsLeaf() const {
				return children[0] < 0 && children[1] < 0;
			}
			
			std::array<int, 2> children;
			unsigned int begin;
			unsigned int n;
		};

		struct QueueItem {
			unsigned int n, d;
		};

		using TraversalPredicate = std::function<bool(unsigned int nodeIndex, unsigned int depth)>;
		using TraversalCallback = std::function <void(unsigned int nodeIndex, unsigned int depth)>;
		using TraversalPriorityLess = std::function<bool(std::array<int, 2> const& nodes)>;
		using TraversalQueue = std::queue<QueueItem>;

		Tree(const unsigned int n)
			: m_List(n)
		{}

		virtual ~Tree() = default;

		[[nodiscard]]
		Node const& GetNode(unsigned int i) const {
			return m_Nodes[i];
		}

		[[nodiscard]]
		T const& GetType(unsigned int i) const { 
			return m_Types[i];
		}

		[[nodiscard]]
		unsigned int GetEntity(unsigned int i) const { 
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
			for (unsigned int i = 0u; i < m_List.size(); ++i) {
				box.Extend(GetEntityPosition(i));
			}

			auto ni = AddNode(0, static_cast<unsigned int>(m_List.size()));
			Construct(ni, box, 0, static_cast<unsigned int>(m_List.size()));
		}

		void Update() {
			TraverseDepthFirst(
				[&](unsigned int, unsigned int)
				{
					return true;
				},

				[&](const unsigned int nodeIndex, unsigned int)
				{
					auto const& nd = GetNode(nodeIndex);
					Calculate(nd.begin, nd.n, GetType(nodeIndex));
				}
			);
		}

		void TraverseDepthFirst(const TraversalPredicate predicate, const TraversalCallback callback, TraversalPriorityLess const& priority = nullptr) const {
			if (m_Nodes.empty()) {
				return;
			}

			if (predicate(0, 0)) {
				TraverseDepthFirst(0, 0, predicate, callback, priority);
			}
		}

		void TraverseBreadthFirst(TraversalPredicate const& predicate, TraversalCallback const& callback, unsigned int startNode = 0, TraversalPriorityLess const& priority = nullptr, TraversalQueue& pending = TraversalQueue()) const {
			callback(startNode, 0);

			if (predicate(startNode, 0)) {
				pending.push({ startNode, 0 });
			}

			TraverseBreadthFirst(pending, predicate, callback, priority);
		}
	protected:
		void Construct(unsigned int node, BoundingBox const& box, unsigned int b, unsigned int n) {
			if (n < 10) {
				return;
			}

			int maxDir = 0;
			glm::dvec3 d = box.Diagonal();
			if (d.y >= d.x && d.y >= d.z) {
				maxDir = 1;
			}
			else if (d.z >= d.x && d.z >= d.y) {
				maxDir = 2;
			}

			std::sort(m_List.begin() + b, m_List.begin() + b + n,
				[&](unsigned int a, unsigned int b)
				{
					return GetEntityPosition(a)[maxDir] < GetEntityPosition(b)[maxDir];
				}
			);

			const unsigned int hal = n / 2;
			unsigned int n0 = AddNode(b, hal);
			unsigned int n1 = AddNode(b + hal, n - hal);
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

		void TraverseDepthFirst(unsigned int nodeIndex,const unsigned int depth,const TraversalPredicate predicate,const TraversalCallback callback, TraversalPriorityLess const& priority) const {
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
						pending.push({ static_cast<unsigned int>(node.children[1]), d + 1 });
						pending.push({ static_cast<unsigned int>(node.children[0]), d + 1 });
					}
					else
					{
						pending.push({ static_cast<unsigned int>(node.children[0]), d + 1 });
						pending.push({ static_cast<unsigned int>(node.children[1]), d + 1 });
					}
				}
			}
		}

		unsigned int AddNode(unsigned int b, unsigned int n) {
			T type;
			Calculate(b, n, type);
			m_Types.push_back(type);
			m_Nodes.push_back({ b, n });
			return static_cast<unsigned int>(m_Nodes.size() - 1);
		}

		[[nodiscard]]
		virtual glm::dvec3 const& GetEntityPosition(unsigned int i) const = 0;
		virtual void Calculate(unsigned int b, unsigned int n, T& type) const = 0;
	protected:
		std::vector<unsigned int> m_List;
		std::vector<Node> m_Nodes;
		std::vector<T> m_Types;
	};
}

#endif // !TREE_H
