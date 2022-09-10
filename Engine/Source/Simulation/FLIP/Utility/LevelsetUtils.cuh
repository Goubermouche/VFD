#ifndef LEVELSET_UTILS_CUH
#define LEVELSET_UTILS_CUH

namespace fe {
    template <class T>
    static __host__ __device__ void Swap(T& a, T& b) {
        T c(a);
        a = b;
        b = c;
    }

    template<class T>
    static __host__ __device__ void Sort(T& a, T& b, T& c, T& d)
    {
        if (a > b) { Swap(a, b); }
        if (c > d) { Swap(c, d); }
        if (a > c) { Swap(a, c); }
        if (b > d) { Swap(b, d); }
        if (b > c) { Swap(b, c); }
    }

    template<class T>
    static __host__ __device__ T SortedTetFraction(T phi0, T phi1, T phi2, T phi3) {
        return phi0 * phi0 * phi0 / ((phi0 - phi1) * (phi0 - phi2) * (phi0 - phi3));
    }

    template<class T>
    static __host__ __device__ T SortedPrismFunction(T phi0, T phi1, T phi2, T phi3) {
        T a = phi0 / (phi0 - phi2);
        T b = phi0 / (phi0 - phi3);
        T c = phi1 / (phi1 - phi3);
        T d = phi1 / (phi1 - phi2);
        return a * b * (1 - d) + b * (1 - c) * d + c * d;
    }

    static __host__ __device__ float VolumeFraction(float phi0, float phi1, float phi2, float phi3) {
        Sort(phi0, phi1, phi2, phi3);
        if (phi3 <= 0) {
            return 1;
        }
        else if (phi2 <= 0) {
            return 1 - SortedTetFraction(phi3, phi2, phi1, phi0);
        }
        else if (phi1 <= 0) {
            return SortedPrismFunction(phi0, phi1, phi2, phi3);
        }
        else if (phi0 <= 0) {
            return SortedTetFraction(phi0, phi1, phi2, phi3);
        }
        else {
            return 0;
        }
    }

    static __host__ __device__ float VolumeFraction(float phi000, float phi100, float phi010, float phi110, float phi001, float phi101, float phi011, float phi111) {
        return (VolumeFraction(phi000, phi001, phi101, phi011) +
            VolumeFraction(phi000, phi101, phi100, phi110) +
            VolumeFraction(phi000, phi010, phi011, phi110) +
            VolumeFraction(phi101, phi011, phi111, phi110) +
            2 * VolumeFraction(phi000, phi011, phi101, phi110) +
            VolumeFraction(phi100, phi101, phi001, phi111) +
            VolumeFraction(phi100, phi001, phi000, phi010) +
            VolumeFraction(phi100, phi110, phi111, phi010) +
            VolumeFraction(phi001, phi111, phi011, phi010) +
            2 * VolumeFraction(phi100, phi111, phi001, phi010)) / 12.0f;
    }

    static __host__ __device__ float FractionInside(float phiLeft, float phiRight) {
        if (phiLeft < 0 && phiRight < 0) {
            return 1;
        }
        if (phiLeft < 0 && phiRight >= 0) {
            return phiLeft / (phiLeft - phiRight);
        }
        if (phiLeft >= 0 && phiRight < 0) {
            return phiRight / (phiRight - phiLeft);
        }

        return 0;
    }

    static __host__ __device__ void _cycleArray(float* arr, int size) {
        float t = arr[0];
        for (int i = 0; i < size - 1; ++i) {
            arr[i] = arr[i + 1];
        }
        arr[size - 1] = t;
    }

    static __host__ __device__ float FractionInside(float phibl, float phibr, float phitl, float phitr) {

        int insideCount = (phibl < 0 ? 1 : 0) +
            (phitl < 0 ? 1 : 0) +
            (phibr < 0 ? 1 : 0) +
            (phitr < 0 ? 1 : 0);
        float list[] = { phibl, phibr, phitr, phitl };

        if (insideCount == 4) {
            return 1;
        }
        else if (insideCount == 3) {
            while (list[0] < 0) {
                _cycleArray(list, 4);
            }

            float side0 = 1 - FractionInside(list[0], list[3]);
            float side1 = 1 - FractionInside(list[0], list[1]);
            return 1.0f - 0.5f * side0 * side1;
        }
        else if (insideCount == 2) {
            while (list[0] >= 0 || !(list[1] < 0 || list[2] < 0)) {
                _cycleArray(list, 4);
            }

            if (list[1] < 0) {
                float sideLeft = FractionInside(list[0], list[3]);
                float sideRight = FractionInside(list[1], list[2]);
                return  0.5f * (sideLeft + sideRight);
            }
            else {
                float middlePoint = 0.25f * (list[0] + list[1] + list[2] + list[3]);
                if (middlePoint < 0) {
                    float area = 0;

                    float side1 = 1 - FractionInside(list[0], list[3]);
                    float side3 = 1 - FractionInside(list[2], list[3]);

                    area += 0.5f * side1 * side3;

                    float side2 = 1 - FractionInside(list[2], list[1]);
                    float side0 = 1 - FractionInside(list[0], list[1]);
                    area += 0.5f * side0 * side2;

                    return 1.0f - area;
                }
                else {
                    float area = 0;

                    float side0 = FractionInside(list[0], list[1]);
                    float side1 = FractionInside(list[0], list[3]);
                    area += 0.5f * side0 * side1;

                    float side2 = FractionInside(list[2], list[1]);
                    float side3 = FractionInside(list[2], list[3]);
                    area += 0.5f * side2 * side3;
                    return area;
                }

            }
        }
        else if (insideCount == 1) {
            while (list[0] >= 0) {
                _cycleArray(list, 4);
            }

            float side0 = FractionInside(list[0], list[3]);
            float side1 = FractionInside(list[0], list[1]);
            return 0.5f * side0 * side1;
        }
        else {
            return 0;
        }
    }
}

#endif // !LEVELSET_UTILS_CUH