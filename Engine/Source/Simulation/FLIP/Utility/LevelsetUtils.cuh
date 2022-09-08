#ifndef LEVELSET_UTILS_CUH
#define LEVELSET_UTILS_CUH

namespace fe {
    template <class T>
    static void _swap(T& a, T& b) {
        T c(a);
        a = b;
        b = c;
    }

    template<class T>
    static __host__ void _sort(T& a, T& b, T& c, T& d)
    {
        if (a > b) { _swap(a, b); }
        if (c > d) { _swap(c, d); }
        if (a > c) { _swap(a, c); }
        if (b > d) { _swap(b, d); }
        if (b > c) { _swap(b, c); }
    }

    // Assumes phi0<0 and phi1>=0, phi2>=0, and phi3>=0 or vice versa.
// In particular, phi0 must not equal any of phi1, phi2 or phi3.
    template<class T>
    static __host__ T _sortedTetFraction(T phi0, T phi1, T phi2, T phi3) {
        return phi0 * phi0 * phi0 / ((phi0 - phi1) * (phi0 - phi2) * (phi0 - phi3));
    }

    // Assumes phi0<0, phi1<0, and phi2>=0, and phi3>=0 or vice versa.
    // In particular, phi0 and phi1 must not equal any of phi2 and phi3.
    template<class T>
    static __host__ T _sortedPrismFraction(T phi0, T phi1, T phi2, T phi3) {
        T a = phi0 / (phi0 - phi2);
        T b = phi0 / (phi0 - phi3);
        T c = phi1 / (phi1 - phi3);
        T d = phi1 / (phi1 - phi2);
        return a * b * (1 - d) + b * (1 - c) * d + c * d;
    }

    static __host__ float volumeFraction(float phi0, float phi1, float phi2, float phi3) {
        _sort(phi0, phi1, phi2, phi3);
        if (phi3 <= 0) {
            return 1;
        }
        else if (phi2 <= 0) {
            return 1 - _sortedTetFraction(phi3, phi2, phi1, phi0);
        }
        else if (phi1 <= 0) {
            return _sortedPrismFraction(phi0, phi1, phi2, phi3);
        }
        else if (phi0 <= 0) {
            return _sortedTetFraction(phi0, phi1, phi2, phi3);
        }
        else {
            return 0;
        }
    }

    static __host__ float VolumeFraction(float phi000, float phi100,
        float phi010, float phi110,
        float phi001, float phi101,
        float phi011, float phi111) {
        // This is the average of the two possible decompositions of the cube into
        // five tetrahedra.
        return (volumeFraction(phi000, phi001, phi101, phi011) +
            volumeFraction(phi000, phi101, phi100, phi110) +
            volumeFraction(phi000, phi010, phi011, phi110) +
            volumeFraction(phi101, phi011, phi111, phi110) +
            2 * volumeFraction(phi000, phi011, phi101, phi110) +
            volumeFraction(phi100, phi101, phi001, phi111) +
            volumeFraction(phi100, phi001, phi000, phi010) +
            volumeFraction(phi100, phi110, phi111, phi010) +
            volumeFraction(phi001, phi111, phi011, phi010) +
            2 * volumeFraction(phi100, phi111, phi001, phi010)) / 12.0f;
    }

    //Given two signed distance values (line endpoints), determine what fraction of a connecting segment is "inside"
    static __host__ float FractionInside(float phiLeft, float phiRight) {
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

    static __host__ void _cycleArray(float* arr, int size) {
        float t = arr[0];
        for (int i = 0; i < size - 1; ++i) {
            arr[i] = arr[i + 1];
        }
        arr[size - 1] = t;
    }


    //Given four signed distance values (square corners), determine what fraction of the square is "inside"
    static __host__ float FractionInside(float phibl, float phibr, float phitl, float phitr) {

        int insideCount = (phibl < 0 ? 1 : 0) +
            (phitl < 0 ? 1 : 0) +
            (phibr < 0 ? 1 : 0) +
            (phitr < 0 ? 1 : 0);
        float list[] = { phibl, phibr, phitr, phitl };

        if (insideCount == 4) {
            return 1;
        }
        else if (insideCount == 3) {
            //rotate until the positive value is in the first position
            while (list[0] < 0) {
                _cycleArray(list, 4);
            }

            //Work out the area of the exterior triangle
            float side0 = 1 - FractionInside(list[0], list[3]);
            float side1 = 1 - FractionInside(list[0], list[1]);
            return 1.0f - 0.5f * side0 * side1;
        }
        else if (insideCount == 2) {

            //rotate until a negative value is in the first position, and the next negative is in either slot 1 or 2.
            while (list[0] >= 0 || !(list[1] < 0 || list[2] < 0)) {
                _cycleArray(list, 4);
            }

            if (list[1] < 0) { //the matching signs are adjacent
                float sideLeft = FractionInside(list[0], list[3]);
                float sideRight = FractionInside(list[1], list[2]);
                return  0.5f * (sideLeft + sideRight);
            }
            else {
                //matching signs are diagonally opposite
                //determine the centre point's sign to disambiguate this case
                float middlePoint = 0.25f * (list[0] + list[1] + list[2] + list[3]);
                if (middlePoint < 0) {
                    float area = 0;

                    //first triangle (top left)
                    float side1 = 1 - FractionInside(list[0], list[3]);
                    float side3 = 1 - FractionInside(list[2], list[3]);

                    area += 0.5f * side1 * side3;

                    //second triangle (top right)
                    float side2 = 1 - FractionInside(list[2], list[1]);
                    float side0 = 1 - FractionInside(list[0], list[1]);
                    area += 0.5f * side0 * side2;

                    return 1.0f - area;
                }
                else {
                    float area = 0;

                    //first triangle (bottom left)
                    float side0 = FractionInside(list[0], list[1]);
                    float side1 = FractionInside(list[0], list[3]);
                    area += 0.5f * side0 * side1;

                    //second triangle (top right)
                    float side2 = FractionInside(list[2], list[1]);
                    float side3 = FractionInside(list[2], list[3]);
                    area += 0.5f * side2 * side3;
                    return area;
                }

            }
        }
        else if (insideCount == 1) {
            //rotate until the negative value is in the first position
            while (list[0] >= 0) {
                _cycleArray(list, 4);
            }

            //Work out the area of the interior triangle, and subtract from 1.
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
