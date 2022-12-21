#ifndef SCALAR_3_F_8_H
#define SCALAR_3_F_8_H

#include "Core/Math/Scalar8.h"

// Inspired by the AVX math library

namespace vfd {
	class Scalar3f8
	{
	public:
		Scalar3f8() = default;
		Scalar3f8(const glm::vec3& x) {
			v[0].v = _mm256_set1_ps(x[0]);
			v[1].v = _mm256_set1_ps(x[1]);
			v[2].v = _mm256_set1_ps(x[2]);
		}

		Scalar8& x() { 
			return v[0];
		}

		Scalar8& y() { 
			return v[1]; 
		}

		Scalar8& z() { 
			return v[2];
		}

		const Scalar8& x() const { 
			return v[0]; 
		}

		const Scalar8& y() const {
			return v[1]; 
		}

		const Scalar8& z() const {
			return v[2]; 
		}

		Scalar8 Norm() const
		{
			return _mm256_sqrt_ps(_mm256_fmadd_ps(v[0].v, v[0].v, _mm256_fmadd_ps(v[1].v, v[1].v, _mm256_mul_ps(v[2].v, v[2].v))));
		}

		Scalar8 SquaredNorm() const {
			return _mm256_fmadd_ps(v[0].v, v[0].v, _mm256_fmadd_ps(v[1].v, v[1].v, _mm256_mul_ps(v[2].v, v[2].v)));
		}

		Scalar8 Dot(const Scalar3f8& a) const {
			Scalar8 res;
			res.v = _mm256_fmadd_ps(v[0].v, a.v[0].v, _mm256_fmadd_ps(v[1].v, a.v[1].v, _mm256_mul_ps(v[2].v, a.v[2].v)));
			return res;
		}

		const Scalar8& operator [] (int i) const { 
			return v[i]; 
		}

		void SetZero() {
			v[0].v = _mm256_setzero_ps(); 
			v[1].v = _mm256_setzero_ps();
			v[2].v = _mm256_setzero_ps();
		}
	public:
		Scalar8 v[3];
	};

	inline Scalar3f8 operator - (Scalar3f8 const& a, Scalar3f8 const& b) {
		Scalar3f8 res;
		res.v[0].v = _mm256_sub_ps(a[0].v, b[0].v);
		res.v[1].v = _mm256_sub_ps(a[1].v, b[1].v);
		res.v[2].v = _mm256_sub_ps(a[2].v, b[2].v);
		return res;
	}

	inline Scalar3f8 operator + (Scalar3f8 const& a, Scalar3f8 const& b) {
		Scalar3f8 res;
		res.v[0].v = _mm256_add_ps(a[0].v, b[0].v);
		res.v[1].v = _mm256_add_ps(a[1].v, b[1].v);
		res.v[2].v = _mm256_add_ps(a[2].v, b[2].v);
		return res;
	}

	inline Scalar3f8 operator * (Scalar3f8 const& a, const Scalar8& s) {
		Scalar3f8 res;
		res.v[0].v = _mm256_mul_ps(a[0].v, s.v);
		res.v[1].v = _mm256_mul_ps(a[1].v, s.v);
		res.v[2].v = _mm256_mul_ps(a[2].v, s.v);
		return res;
	}

	inline Scalar3f8 ConvertScalarZero(const unsigned int* idx, const float* v, const unsigned char count = 8u)
	{
		Scalar3f8 x;
		switch (count)
		{
		case 1u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 2u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], v[3 * idx[1] + 0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], v[3 * idx[1] + 1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], v[3 * idx[1] + 2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 3u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 4u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 5u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], 0.0f, 0.0f, 0.0f);
			break;
		case 6u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], v[3 * idx[5] + 0], 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], v[3 * idx[5] + 1], 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], v[3 * idx[5] + 2], 0.0f, 0.0f);
			break;
		case 7u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], v[3 * idx[5] + 0], v[3 * idx[6] + 0], 0.0f);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], v[3 * idx[5] + 1], v[3 * idx[6] + 1], 0.0f);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], v[3 * idx[5] + 2], v[3 * idx[6] + 2], 0.0f);
			break;
		case 8u:
			x.v[0].v = _mm256_setr_ps(v[3 * idx[0] + 0], v[3 * idx[1] + 0], v[3 * idx[2] + 0], v[3 * idx[3] + 0], v[3 * idx[4] + 0], v[3 * idx[5] + 0], v[3 * idx[6] + 0], v[3 * idx[7] + 0]);
			x.v[1].v = _mm256_setr_ps(v[3 * idx[0] + 1], v[3 * idx[1] + 1], v[3 * idx[2] + 1], v[3 * idx[3] + 1], v[3 * idx[4] + 1], v[3 * idx[5] + 1], v[3 * idx[6] + 1], v[3 * idx[7] + 1]);
			x.v[2].v = _mm256_setr_ps(v[3 * idx[0] + 2], v[3 * idx[1] + 2], v[3 * idx[2] + 2], v[3 * idx[3] + 2], v[3 * idx[4] + 2], v[3 * idx[5] + 2], v[3 * idx[6] + 2], v[3 * idx[7] + 2]);
		}
		return x;
	}

	inline Scalar3f8 ConvertScalarZero(const unsigned int* idx, const glm::vec3* v, const unsigned char count = 8u) {
		Scalar3f8 x;
		switch (count)
		{
		case 1u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 2u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], v[idx[1]][0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], v[idx[1]][1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], v[idx[1]][2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 3u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 4u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], 0.0f, 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], 0.0f, 0.0f, 0.0f, 0.0f);
			break;
		case 5u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], 0.0f, 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], 0.0f, 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], 0.0f, 0.0f, 0.0f);
			break;
		case 6u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], v[idx[5]][0], 0.0f, 0.0f);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], v[idx[5]][1], 0.0f, 0.0f);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], v[idx[5]][2], 0.0f, 0.0f);
			break;
		case 7u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], v[idx[5]][0], v[idx[6]][0], 0.0f);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], v[idx[5]][1], v[idx[6]][1], 0.0f);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], v[idx[5]][2], v[idx[6]][2], 0.0f);
			break;
		case 8u:
			x.v[0].v = _mm256_setr_ps(v[idx[0]][0], v[idx[1]][0], v[idx[2]][0], v[idx[3]][0], v[idx[4]][0], v[idx[5]][0], v[idx[6]][0], v[idx[7]][0]);
			x.v[1].v = _mm256_setr_ps(v[idx[0]][1], v[idx[1]][1], v[idx[2]][1], v[idx[3]][1], v[idx[4]][1], v[idx[5]][1], v[idx[6]][1], v[idx[7]][1]);
			x.v[2].v = _mm256_setr_ps(v[idx[0]][2], v[idx[1]][2], v[idx[2]][2], v[idx[3]][2], v[idx[4]][2], v[idx[5]][2], v[idx[6]][2], v[idx[7]][2]);
		}
		
		return x;
	}
}

#endif // !SCALAR_3_F_8_H