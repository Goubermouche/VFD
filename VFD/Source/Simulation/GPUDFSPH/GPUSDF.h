#ifndef GPU_SDF_H
#define GPU_SDF_H

#include "Renderer/Mesh/TriangleMesh.h"
#include <Core/Structures/BoundingBox.h>

namespace vfd {
	struct Array1True {};
	struct Array1False {};
	template<typename T> struct Array1IsIntegral { typedef Array1False type; }; // default: no (specializations to yes follow)
	template<> struct Array1IsIntegral<bool> { typedef Array1True type; };
	template<> struct Array1IsIntegral<char> { typedef Array1True type; };
	template<> struct Array1IsIntegral<signed char> { typedef Array1True type; };
	template<> struct Array1IsIntegral<unsigned char> { typedef Array1True type; };
	template<> struct Array1IsIntegral<short> { typedef Array1True type; };
	template<> struct Array1IsIntegral<unsigned short> { typedef Array1True type; };
	template<> struct Array1IsIntegral<int> { typedef Array1True type; };
	template<> struct Array1IsIntegral<unsigned int> { typedef Array1True type; };
	template<> struct Array1IsIntegral<long> { typedef Array1True type; };
	template<> struct Array1IsIntegral<unsigned long> { typedef Array1True type; };
	template<> struct Array1IsIntegral<long long> { typedef Array1True type; };
	template<> struct Array1IsIntegral<unsigned long long> { typedef Array1True type; };

	template<typename T>
	struct Array1D {
typedef T* iterator;
		typedef const T* const_iterator;
		typedef unsigned long size_type;
		typedef long difference_type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T value_type;
		typedef T* pointer;
		typedef const T* const_pointer;		
		typedef std::reverse_iterator<iterator> reverse_iterator;
		typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

		unsigned long n;
		unsigned long max_n;
		T* data;

		Array1D(void)
			: n(0), max_n(0), data(0)
		{}

		// note: default initial values are zero
		Array1D(unsigned long n_)
			: n(0), max_n(0), data(0)
		{
			if (n_ > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
			data = (T*)std::calloc(n_, sizeof(T));
			if (!data) throw std::bad_alloc();
			n = n_;
			max_n = n_;
		}

		Array1D(unsigned long n_, const T& value)
			: n(0), max_n(0), data(0)
		{
			if (n_ > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
			data = (T*)std::calloc(n_, sizeof(T));
			if (!data) throw std::bad_alloc();
			n = n_;
			max_n = n_;
			for (unsigned long i = 0; i < n; ++i) data[i] = value;
		}

		Array1D(unsigned long n_, const T& value, unsigned long max_n_)
			: n(0), max_n(0), data(0)
		{
			assert(n_ <= max_n_);
			if (max_n_ > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
			data = (T*)std::calloc(max_n_, sizeof(T));
			if (!data) throw std::bad_alloc();
			n = n_;
			max_n = max_n_;
			for (unsigned long i = 0; i < n; ++i) data[i] = value;
		}

		Array1D(unsigned long n_, const T* data_)
			: n(0), max_n(0), data(0)
		{
			if (n_ > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
			data = (T*)std::calloc(n_, sizeof(T));
			if (!data) throw std::bad_alloc();
			n = n_;
			max_n = n_;
			assert(data_);
			std::memcpy(data, data_, n * sizeof(T));
		}

		Array1D(unsigned long n_, const T* data_, unsigned long max_n_)
			: n(0), max_n(0), data(0)
		{
			assert(n_ <= max_n_);
			if (max_n_ > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
			data = (T*)std::calloc(max_n_, sizeof(T));
			if (!data) throw std::bad_alloc();
			max_n = max_n_;
			n = n_;
			assert(data_);
			std::memcpy(data, data_, n * sizeof(T));
		}

		Array1D(const Array1D<T>& x)
			: n(0), max_n(0), data(0)
		{
			data = (T*)std::malloc(x.n * sizeof(T));
			if (!data) throw std::bad_alloc();
			n = x.n;
			max_n = x.n;
			std::memcpy(data, x.data, n * sizeof(T));
		}

		~Array1D(void)
		{
			std::free(data);
#ifndef NDEBUG
			data = 0;
			n = max_n = 0;
#endif
		}

		const T& operator[](unsigned long i) const
		{
			return data[i];
		}

		T& operator[](unsigned long i)
		{
			return data[i];
		}

		// these are range-checked (in debug mode) versions of operator[], like at()
		const T& operator()(unsigned long i) const
		{
			assert(i < n);
			return data[i];
		}

		T& operator()(unsigned long i)
		{
			assert(i < n);
			return data[i];
		}

		Array1D<T>& operator=(const Array1D<T>& x)
		{
			if (max_n < x.n) {
				T* new_data = (T*)std::malloc(x.n * sizeof(T));
				if (!new_data) throw std::bad_alloc();
				std::free(data);
				data = new_data;
				max_n = x.n;
			}
			n = x.n;
			std::memcpy(data, x.data, n * sizeof(T));
			return *this;
		}

		bool operator==(const Array1D<T>& x) const
		{
			if (n != x.n) return false;
			for (unsigned long i = 0; i < n; ++i) if (!(data[i] == x.data[i])) return false;
			return true;
		}

		bool operator!=(const Array1D<T>& x) const
		{
			if (n != x.n) return true;
			for (unsigned long i = 0; i < n; ++i) if (data[i] != x.data[i]) return true;
			return false;
		}

		bool operator<(const Array1D<T>& x) const
		{
			for (unsigned long i = 0; i < n && i < x.n; ++i) {
				if (data[i] < x[i]) return true;
				else if (x[i] < data[i]) return false;
			}
			return n < x.n;
		}

		bool operator>(const Array1D<T>& x) const
		{
			for (unsigned long i = 0; i < n && i < x.n; ++i) {
				if (data[i] > x[i]) return true;
				else if (x[i] > data[i]) return false;
			}
			return n > x.n;
		}

		bool operator<=(const Array1D<T>& x) const
		{
			for (unsigned long i = 0; i < n && i < x.n; ++i) {
				if (data[i] < x[i]) return true;
				else if (x[i] < data[i]) return false;
			}
			return n <= x.n;
		}

		bool operator>=(const Array1D<T>& x) const
		{
			for (unsigned long i = 0; i < n && i < x.n; ++i) {
				if (data[i] > x[i]) return true;
				else if (x[i] > data[i]) return false;
			}
			return n >= x.n;
		}

		void add_unique(const T& value)
		{
			for (unsigned long i = 0; i < n; ++i) if (data[i] == value) return;
			if (n == max_n) grow();
			data[n++] = value;
		}

		void assign(const T& value)
		{
			for (unsigned long i = 0; i < n; ++i) data[i] = value;
		}

		void assign(unsigned long num, const T& value)
		{
			fill(num, value);
		}

		// note: copydata may not alias this array's data, and this should not be
		// used when T is a full object (which defines its own copying operation)
		void assign(unsigned long num, const T* copydata)
		{
			assert(num == 0 || copydata);
			if (num > max_n) {
				if (num > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
				std::free(data);
				data = (T*)std::malloc(num * sizeof(T));
				if (!data) throw std::bad_alloc();
				max_n = num;
			}
			n = num;
			std::memcpy(data, copydata, n * sizeof(T));
		}

		template<typename InputIterator>
		void assign(InputIterator first, InputIterator last)
		{
			assign_(first, last, typename Array1IsIntegral<InputIterator>::type());
		}

		template<typename InputIterator>
		void assign_(InputIterator first, InputIterator last, Array1True check)
		{
			fill(first, last);
		}

		template<typename InputIterator>
		void assign_(InputIterator first, InputIterator last, Array1False check)
		{
			unsigned long i = 0;
			InputIterator p = first;
			for (; p != last; ++p, ++i) {
				if (i == max_n) grow();
				data[i] = *p;
			}
			n = i;
		}

		const T& at(unsigned long i) const
		{
			assert(i < n);
			return data[i];
		}

		T& at(unsigned long i)
		{
			assert(i < n);
			return data[i];
		}

		const T& back(void) const
		{
			assert(data && n > 0);
			return data[n - 1];
		}

		T& back(void)
		{
			assert(data && n > 0);
			return data[n - 1];
		}

		const T* begin(void) const
		{
			return data;
		}

		T* begin(void)
		{
			return data;
		}

		unsigned long capacity(void) const
		{
			return max_n;
		}

		void clear(void)
		{
			std::free(data);
			data = 0;
			max_n = 0;
			n = 0;
		}

		bool empty(void) const
		{
			return n == 0;
		}

		const T* end(void) const
		{
			return data + n;
		}

		T* end(void)
		{
			return data + n;
		}

		void erase(unsigned long index)
		{
			assert(index < n);
			for (unsigned long i = index; i < n - 1; ++i)
				data[i] = data[i - 1];
			pop_back();
		}

		void fill(unsigned long num, const T& value)
		{
			if (num > max_n) {
				if (num > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
				std::free(data);
				data = (T*)std::malloc(num * sizeof(T));
				if (!data) throw std::bad_alloc();
				max_n = num;
			}
			n = num;
			for (unsigned long i = 0; i < n; ++i) data[i] = value;
		}

		const T& front(void) const
		{
			assert(n > 0);
			return *data;
		}

		T& front(void)
		{
			assert(n > 0);
			return *data;
		}

		void grow(void)
		{
			unsigned long new_size = (max_n * sizeof(T) < ULONG_MAX / 2 ? 2 * max_n + 1 : ULONG_MAX / sizeof(T));
			T* new_data = (T*)std::realloc(data, new_size * sizeof(T));
			if (!new_data) throw std::bad_alloc();
			data = new_data;
			max_n = new_size;
		}

		void insert(unsigned long index, const T& entry)
		{
			assert(index <= n);
			push_back(back());
			for (unsigned long i = n - 1; i > index; --i)
				data[i] = data[i - 1];
			data[index] = entry;
		}

		unsigned long max_size(void) const
		{
			return ULONG_MAX / sizeof(T);
		}

		void pop_back(void)
		{
			assert(n > 0);
			--n;
		}

		void push_back(const T& value)
		{
			if (n == max_n) grow();
			data[n++] = value;
		}

		reverse_iterator rbegin(void)
		{
			return reverse_iterator(end());
		}

		const_reverse_iterator rbegin(void) const
		{
			return const_reverse_iterator(end());
		}

		reverse_iterator rend(void)
		{
			return reverse_iterator(begin());
		}

		const_reverse_iterator rend(void) const
		{
			return const_reverse_iterator(begin());
		}

		void reserve(unsigned long r)
		{
			if (r > ULONG_MAX / sizeof(T)) throw std::bad_alloc();
			T* new_data = (T*)std::realloc(data, r * sizeof(T));
			if (!new_data) throw std::bad_alloc();
			data = new_data;
			max_n = r;
		}

		void resize(unsigned long n_)
		{
			if (n_ > max_n) reserve(n_);
			n = n_;
		}

		void resize(unsigned long n_, const T& value)
		{
			if (n_ > max_n) reserve(n_);
			if (n < n_) for (unsigned long i = n; i < n_; ++i) data[i] = value;
			n = n_;
		}

		void set_zero(void)
		{
			std::memset(data, 0, n * sizeof(T));
		}

		unsigned long size(void) const
		{
			return n;
		}

		void swap(Array1D<T>& x)
		{
			std::swap(n, x.n);
			std::swap(max_n, x.max_n);
			std::swap(data, x.data);
		}

		// resize the array to avoid wasted space, without changing contents
		// (Note: realloc, at least on some platforms, will not do the trick)
		void trim(void)
		{
			if (n == max_n) return;
			T* new_data = (T*)std::malloc(n * sizeof(T));
			if (!new_data) return;
			std::memcpy(new_data, data, n * sizeof(T));
			std::free(data);
			data = new_data;
			max_n = n;
		}
	};

    template<class T, class ArrayT = std::vector<T>>
	struct Array3D {
		typedef typename ArrayT::iterator iterator;
		typedef typename ArrayT::const_iterator const_iterator;
		typedef typename ArrayT::size_type size_type;
		typedef long difference_type;
		typedef T& reference;
		typedef const T& const_reference;
		typedef T value_type;
		typedef T* pointer;
		typedef const T* const_pointer;
		typedef typename ArrayT::reverse_iterator reverse_iterator;
		typedef typename ArrayT::const_reverse_iterator const_reverse_iterator;

		int ni, nj, nk;
		ArrayT a;

		Array3D(void)
			: ni(0), nj(0), nk(0)
		{}

		Array3D(int ni_, int nj_, int nk_)
			: ni(ni_), nj(nj_), nk(nk_), a(ni_* nj_* nk_)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		}

		Array3D(int ni_, int nj_, int nk_, ArrayT& a_)
			: ni(ni_), nj(nj_), nk(nk_), a(a_)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		}

		Array3D(int ni_, int nj_, int nk_, const T& value)
			: ni(ni_), nj(nj_), nk(nk_), a(ni_* nj_* nk_, value)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		}

		Array3D(int ni_, int nj_, int nk_, const T& value, size_type max_n_)
			: ni(ni_), nj(nj_), nk(nk_), a(ni_* nj_* nk_, value, max_n_)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		}

		Array3D(int ni_, int nj_, int nk_, T* data_)
			: ni(ni_), nj(nj_), nk(nk_), a(ni_* nj_* nk_, data_)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		}

		Array3D(int ni_, int nj_, int nk_, T* data_, size_type max_n_)
			: ni(ni_), nj(nj_), nk(nk_), a(ni_* nj_* nk_, data_, max_n_)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
		}

		~Array3D(void)
		{
#ifndef NDEBUG
			ni = nj = 0;
#endif
		}

		const T& operator()(int i, int j, int k) const
		{
			assert(i >= 0 && i < ni&& j >= 0 && j < nj&& k >= 0 && k < nk);
			return a[i + ni * (j + nj * k)];
		}

		T& operator()(int i, int j, int k)
		{
			assert(i >= 0 && i < ni&& j >= 0 && j < nj&& k >= 0 && k < nk);
			return a[i + ni * (j + nj * k)];
		}

		T operator [] (int i) const { return a[i]; }
		T& operator [] (int i) { return a[i]; }


		bool operator==(const Array3D<T>& x) const
		{
			return ni == x.ni && nj == x.nj && nk == x.nk && a == x.a;
		}

		bool operator!=(const Array3D<T>& x) const
		{
			return ni != x.ni || nj != x.nj || nk != x.nk || a != x.a;
		}

		bool operator<(const Array3D<T>& x) const
		{
			if (ni < x.ni) return true; else if (ni > x.ni) return false;
			if (nj < x.nj) return true; else if (nj > x.nj) return false;
			if (nk < x.nk) return true; else if (nk > x.nk) return false;
			return a < x.a;
		}

		bool operator>(const Array3D<T>& x) const
		{
			if (ni > x.ni) return true; else if (ni < x.ni) return false;
			if (nj > x.nj) return true; else if (nj < x.nj) return false;
			if (nk > x.nk) return true; else if (nk < x.nk) return false;
			return a > x.a;
		}

		bool operator<=(const Array3D<T>& x) const
		{
			if (ni < x.ni) return true; else if (ni > x.ni) return false;
			if (nj < x.nj) return true; else if (nj > x.nj) return false;
			if (nk < x.nk) return true; else if (nk > x.nk) return false;
			return a <= x.a;
		}

		bool operator>=(const Array3D<T>& x) const
		{
			if (ni > x.ni) return true; else if (ni < x.ni) return false;
			if (nj > x.nj) return true; else if (nj < x.nj) return false;
			if (nk > x.nk) return true; else if (nk < x.nk) return false;
			return a >= x.a;
		}

		void assign(const T& value)
		{
			a.assign(value);
		}

		void assign(int ni_, int nj_, int nk_, const T& value)
		{
			a.assign(ni_ * nj_ * nk_, value);
			ni = ni_;
			nj = nj_;
			nk = nk_;
		}

		void assign(int ni_, int nj_, int nk_, const T* copydata)
		{
			a.assign(ni_ * nj_ * nk_, copydata);
			ni = ni_;
			nj = nj_;
			nk = nk_;
		}

		const T& at(int i, int j, int k) const
		{
			assert(i >= 0 && i < ni&& j >= 0 && j < nj&& k >= 0 && k < nk);
			return a[i + ni * (j + nj * k)];
		}

		T& at(int i, int j, int k)
		{
			assert(i >= 0 && i < ni&& j >= 0 && j < nj&& k >= 0 && k < nk);
			return a[i + ni * (j + nj * k)];
		}

		const T& back(void) const
		{
			assert(a.size());
			return a.back();
		}

		T& back(void)
		{
			assert(a.size());
			return a.back();
		}
		
		const_iterator begin(void) const
		{
			return a.begin();
		}

		iterator begin(void)
		{
			return a.begin();
		}

		size_type capacity(void) const
		{
			return a.capacity();
		}

		void clear(void)
		{
			a.clear();
			ni = nj = nk = 0;
		}

		bool empty(void) const
		{
			return a.empty();
		}

		const_iterator end(void) const
		{
			return a.end();
		}

		iterator end(void)
		{
			return a.end();
		}

		void fill(int ni_, int nj_, int nk_, const T& value)
		{
			a.fill(ni_ * nj_ * nk_, value);
			ni = ni_;
			nj = nj_;
			nk = nk_;
		}

		const T& front(void) const
		{
			assert(a.size());
			return a.front();
		}

		T& front(void)
		{
			assert(a.size());
			return a.front();
		}

		size_type max_size(void) const
		{
			return a.max_size();
		}

		reverse_iterator rbegin(void)
		{
			return reverse_iterator(end());
		}

		const_reverse_iterator rbegin(void) const
		{
			return const_reverse_iterator(end());
		}

		reverse_iterator rend(void)
		{
			return reverse_iterator(begin());
		}

		const_reverse_iterator rend(void) const
		{
			return const_reverse_iterator(begin());
		}

		void reserve(int reserve_ni, int reserve_nj, int reserve_nk)
		{
			a.reserve(reserve_ni * reserve_nj * reserve_nk);
		}

		void resize(int ni_, int nj_, int nk_)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
			a.resize(ni_ * nj_ * nk_);
			ni = ni_;
			nj = nj_;
			nk = nk_;
		}

		void resize(int ni_, int nj_, int nk_, const T& value)
		{
			assert(ni_ >= 0 && nj_ >= 0 && nk_ >= 0);
			a.resize(ni_ * nj_ * nk_, value);
			ni = ni_;
			nj = nj_;
			nk = nk_;
		}

		void set_zero(void)
		{
			a.set_zero();
		}

		size_type size(void) const
		{
			return a.size();
		}

		void swap(Array3D<T>& x)
		{
			std::swap(ni, x.ni);
			std::swap(nj, x.nj);
			std::swap(nk, x.nk);
			a.swap(x.a);
		}

		void trim(void)
		{
			a.trim();
		}

		bool indexInRange(int i, int j, int k) {
			return i >= 0 && i < ni&& j >= 0 && j < nj&& k >= 0 && k < nk;
		}

		bool indexInRange(int i) {
			return i >= 0 && i < ni * nj * nk;
		}
	};

	typedef Array3D<float, Array1D<float>> Array3f;
	typedef Array3D<int, Array1D<int> > Array3i;

	class GPUSDF : public RefCounted
	{
	public:
		GPUSDF(const Ref<TriangleMesh>& mesh);

		// Sampling methods 
		float GetDistance(const glm::vec3& point);
		float GetDistanceTrilinear(const glm::vec3& point);
		float GetDistanceTricubic(const glm::vec3& point);

		const BoundingBox<glm::vec3>& GetDomain();
	private:
		void MakeLevelSet3D(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, const glm::vec3& origin, float dx, Array3f& phi, const int exact_band = 1);

		static float PointToTriangleDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& x3);
		static float PointToSegmentDistance(const glm::vec3& x0, const glm::vec3& x1, const glm::vec3& x2);
		static bool PointInTriangle2D(float x0, float y0,	float x1, float y1, float x2, float y2, float x3, float y3, float& a, float& b, float& c);
		static int Orientation(float x1, float y1, float x2, float y2, float& twice_signed_area);
		static void Sweep(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3f& phi, Array3i& closest_tri, const glm::vec3& origin, float dx, int di, int dj, int dk);
		static void CheckNeighbor(const std::vector<glm::uvec3>& tri, const std::vector<glm::vec3>& x, Array3f& phi, Array3i& closest_tri, const glm::vec3& gx, int i0, int j0, int k0, int i1, int j1, int k1);
	private:
		int padding = 10.0f;

		float m_CellSize = 0.05f;
		float m_CellSizeInverse;
		unsigned int m_CellCount;
		glm::ivec3 m_Resolution;

		Array3f m_PHI;
		BoundingBox<glm::vec3> m_Domain;
	};
}

#endif // !GPU_SDF_H