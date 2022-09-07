#ifndef PRESSURE_SOLVER_CUH
#define PRESSURE_SOLVER_CUH

#include "Simulation/FLIP/Utility/Array3D.cuh"

namespace fe {
    struct WeightGrid {
        Array3D<float> U;
        Array3D<float> V;
        Array3D<float> W;

        __host__ WeightGrid() {}
        __host__ void Init(int i, int j, int k) {
            U.Init(i + 1, j, k, 0.0f);
            V.Init(i, j + 1, k, 0.0f);
            W.Init(i, j, k + 1, 0.0f);
        }

        __host__ void HostFree() {
            U.HostFree();
            V.HostFree();
            W.HostFree();
        }
    };

}

#endif // !PRESSURE_SOLVER_CUH
