#include <string>

using namespace std;

string kernel_code = R"(
        __kernel void determinant(__global double* mat, __global double* result, const int n) {
            int i = get_global_id(0);

            double det = 1.0;

            for (int j = i + 1; j < n; ++j) {
                double ratio = mat[j * n + i] / mat[i * n + i];
                for (int k = 0; k < n; ++k) {
                    mat[j * n + k] -= ratio * mat[i * n + k];
                }
            }

            result[i] = mat[i * n + i];
        }
    )";