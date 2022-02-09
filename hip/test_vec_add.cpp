#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include "hip_vec_add.hpp"
#include <climits>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " n" << endl;
        return 0;
    }

    size_t n = atoi(argv[1]);

    cout << "vector_size = " << n << endl;
    srand(time(nullptr));
    std::vector<double> din1, din2, dres;
    std::vector<float> fin1, fin2, fres;
    std::vector<half> hin1, hin2, hres;
    din1.resize(n);
    din2.resize(n);
    dres.resize(n);
    fin1.resize(n);
    fin2.resize(n);
    fres.resize(n);
    hin1.resize(n);
    hin2.resize(n);
    hres.resize(n);

    for (int i = 0; i < n; ++i)
    {
        din1[i] = 1.0 * rand() / INT_MAX;
        din2[i] = 1.0 * rand() / INT_MAX;
        fin1[i] = din1[i];
        fin2[i] = din2[i];
        hin1[i] = __float2half(fin1[i]);
        hin2[i] = __float2half(fin2[i]);
        double d1 = (float)din1[i];
        double h1 = (float)hin1[i];
        //std::cout << "d = " << d1 << ", f = " << fin1[i] << ", h = " << h1 << std::endl;
    }

    bool dret = hip_vec_add(din1, din2, dres);
    bool fret = hip_vec_add(fin1, fin2, fres);
    bool hret = hip_vec_add(hin1, hin2, hres);
    if (not (dret and fret and hret))
    {
        std::cout << "vector add error!" << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        float d = dres[i];
        float f = fres[i];
        float h = hres[i];

        if (fabs(d - f) > 0.01f or fabs(d - h) > 0.01f and fabs(f - h) > 0.01f)
        {
            std::cout << "d[" << i << "] = " << d << ", f[" << i << "] = " << f << ", h[" << i << "] = " << h << std::endl;
        }
    }

    cout << "PASSED" << endl;

    return 0;
}

