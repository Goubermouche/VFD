#ifndef CACHE_H_
#define CACHE_H_

namespace fe {
	template<typename Key, typename Value>
	class Cache {
	public:
		Cache(std::function<Value(const Key&)>  const& f, std::size_t c)
			: m_EvaluationFunction(f), m_Capacity(c)
		{
			assert(m_Capacity != 0);
		}

		Value operator()(const Key& key) {
			auto it = m_KeyToValue.find(key);

			if (it == m_KeyToValue.end()) {
				auto value = m_EvaluationFunction(key);
				Insert(key, value);
				return value;
			}
			else {
				m_KeyTracker.splice(m_KeyTracker.end(), m_KeyTracker, (*it).second.second);
				return(*it).second.first;
			}
		}
	private:
		void Insert(Key const& key, Value const& value) {
			if (m_KeyToValue.size() == m_Capacity) {
				Evict();
			}

			auto it = m_KeyTracker.insert(m_KeyTracker.end(), key);
			m_KeyToValue.insert(std::make_pair(key, std::make_pair(value, it)));
		}

		void Evict() {
			assert(!m_KeyTracker.empty());
			auto it = m_KeyToValue.find(m_KeyTracker.front());
			assert(it != m_KeyToValue.end());
			m_KeyToValue.erase(it);
			m_KeyTracker.pop_front();
		}
	private:
		std::function<Value(const Key&)> m_EvaluationFunction;
		uint32_t m_Capacity;
		std::list<Key> m_KeyTracker;
		std::map<Key, std::pair<Value, typename std::list<Key>::iterator>> m_KeyToValue;
	};
}

#endif // !CACHE_H_