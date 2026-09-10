#include <gtest/gtest.h>
#include "../include/boundary.hpp"
#include "../include/mesh.hpp"
#include "../include/solver.hpp"

// Compare two vectors with a small tolerance
static void expect_allclose(const std::vector<double>& a,
                            const std::vector<double>& b,
                            double tol = 1e-12) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_NEAR(a[i], b[i], tol) << "Mismatch at index " << i;
    }
}

TEST(Jacobi, N100_Iter1_SerialVsParallel) {
    const int N = 100;
    const int steps = 1;
    const double corner = 1.0;
    const double fill = 0.0;

    // Serial (reference)
    CMesh<double> ms(N, boundary_condition<double>, corner, fill);
    CSolver<double> sol;
    sol.jacobi(ms, steps, /*PrintInterval=*/0, /*run_pll=*/false);

    // Parallel (system under test)
    CMesh<double> mp(N, boundary_condition<double>, corner, fill);
    sol.jacobi(mp, steps, /*PrintInterval=*/0, /*run_pll=*/true);

    expect_allclose(ms.field, mp.field, 1e-12);
}

TEST(Jacobi, N100_Iter1000_SerialVsParallel) {
    const int N = 100;
    const int steps = 1000;
    const double corner = 1.0;
    const double fill = 0.0;

    CMesh<double> ms(N, boundary_condition<double>, corner, fill);
    CSolver<double> sol;
    sol.jacobi(ms, steps, /*PrintInterval=*/0, /*run_pll=*/false);

    CMesh<double> mp(N, boundary_condition<double>, corner, fill);
    sol.jacobi(mp, steps, /*PrintInterval=*/0, /*run_pll=*/true);

    // If tiny numerical drift appears, relax tol to 1e-10 or 1e-9
    expect_allclose(ms.field, mp.field, 1e-12);
}
