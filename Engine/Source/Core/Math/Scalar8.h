#ifndef SCALAR_8_H
#define SCALAR_8_H

// Inspired by the AVX math library

namespace fe {
	class Scalar8
	{
	public:
		Scalar8() = default;

		Scalar8(float f) {
			v = _mm256_set1_ps(f);
		}

		Scalar8(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
			v = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
		}

		Scalar8(float const* p)
		{
			v = _mm256_loadu_ps(p);
		}

		Scalar8(__m256 const& x) {
			v = x;
		}

		Scalar8& operator = (__m256 const& x) {
			v = x;
			return *this;
		}

		void Store(float* p) const {
			_mm256_storeu_ps(p, v);
		}

		inline float Reduce() const {
			const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
			const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
			const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
			return _mm_cvtss_f32(x32);
		}
	public:
		__m256 v;
	};
	
	inline Scalar8 operator - (Scalar8& a) {
		return _mm256_sub_ps(_mm256_set1_ps(0.0), a.v);
	}

	static inline Scalar8 operator - (Scalar8 const& a, Scalar8 const& b) {
		return _mm256_sub_ps(a.v, b.v);
	}

	static inline Scalar8& operator -= (Scalar8& a, Scalar8 const& b) {
		a.v = _mm256_sub_ps(a.v, b.v);
		return a;
	}

	static inline Scalar8 operator + (Scalar8 const& a, Scalar8 const& b) {
		return _mm256_add_ps(a.v, b.v);
	}

	static inline Scalar8& operator += (Scalar8& a, Scalar8 const& b) {
		a.v = _mm256_add_ps(a.v, b.v);
		return a;
	}

	static inline Scalar8 operator * (Scalar8 const& a, Scalar8 const& b) {
		return _mm256_mul_ps(a.v, b.v);
	}

	static inline Scalar8 operator / (Scalar8 const& a, Scalar8 const& b) {
		return _mm256_div_ps(a.v, b.v);
	}

	static inline Scalar8 operator > (Scalar8 const& a, Scalar8 const& b) {
		return _mm256_cmp_ps(b.v, a.v, 1);
	}

	static inline Scalar8 operator <= (Scalar8 const& a, Scalar8 const& b) {
		return _mm256_cmp_ps(a.v, b.v, 2);
	}

	static inline Scalar8 Blend(Scalar8 const& c, Scalar8 const& a, Scalar8 const& b) {
		return _mm256_blendv_ps(b.v, a.v, c.v);
	}

	static inline Scalar8 MultiplyAndSubtract(const Scalar8& a, const Scalar8& b, const Scalar8& c)
	{
		return _mm256_fmsub_ps(a.v, b.v, c.v);
	}

	static inline Scalar8 MultiplyAndAdd(const Scalar8& a, const Scalar8& b, const Scalar8& c)
	{
		return _mm256_fmadd_ps(a.v, b.v, c.v);
	}

	static inline Scalar8 ConvertZero(const unsigned int* idx, const float* x, const unsigned char count = 8u)
	{
		Scalar8 v;
		switch (count)
		{
		case 1u:
			v.v = _mm256_setr_ps(x[idx[0]], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 2u:
			v.v = _mm256_setr_ps(x[idx[0]], x[idx[1]], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 3u:
			v.v = _mm256_setr_ps(x[idx[0]], x[idx[1]], x[idx[2]], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 4u:
			v.v = _mm256_setr_ps(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 5u:
			v.v = _mm256_setr_ps(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], 0.0f, 0.0f, 0.0f); break;
		case 6u:
			v.v = _mm256_setr_ps(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], 0.0f, 0.0f); break;
		case 7u:
			v.v = _mm256_setr_ps(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], x[idx[6]], 0.0f); break;
		case 8u:
			v.v = _mm256_setr_ps(x[idx[0]], x[idx[1]], x[idx[2]], x[idx[3]], x[idx[4]], x[idx[5]], x[idx[6]], x[idx[7]]); break;
		}
		return v;
	}

	static inline Scalar8 ConvertZero(const float x, const unsigned char count = 8u)
	{
		Scalar8 v;
		switch (count)
		{
		case 1u:
			v.v = _mm256_setr_ps(x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 2u:
			v.v = _mm256_setr_ps(x, x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 3u:
			v.v = _mm256_setr_ps(x, x, x, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 4u:
			v.v = _mm256_setr_ps(x, x, x, x, 0.0f, 0.0f, 0.0f, 0.0f); break;
		case 5u:
			v.v = _mm256_setr_ps(x, x, x, x, x, 0.0f, 0.0f, 0.0f); break;
		case 6u:
			v.v = _mm256_setr_ps(x, x, x, x, x, x, 0.0f, 0.0f); break;
		case 7u:
			v.v = _mm256_setr_ps(x, x, x, x, x, x, x, 0.0f); break;
		case 8u:
			v.v = _mm256_setr_ps(x, x, x, x, x, x, x, x); break;
		}
		return v;
	}
}

#endif // !SCALAR_8_H