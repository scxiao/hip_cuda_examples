#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#include "hip_sqrt.hpp"
#include <climits>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " n" << endl;
        return 0;
    }

    int sn = atoi(argv[1]);
    size_t n = (1 << sn);

    cout << "vector_size = " << n << endl;
    srand(time(nullptr));
    std::vector<double> din, dres;
    std::vector<float> fin, fres;
    std::vector<half> hin, hres, h2res;
    din.resize(n);
    dres.resize(n);
    fin.resize(n);
    fres.resize(n);
    hin.resize(n);
    hres.resize(n);

    for (int i = 0; i < n; ++i)
    {
        din[i] = 1.0 * rand() / INT_MAX;
        fin[i] = din[i];
        hin[i] = __float2half(fin[i]);
        double d = (double)din[i];
        double h = (double)hin[i];
    }

    bool dret = hip_sqrt(din, dres);
    bool fret = hip_sqrt(fin, fres);
    bool hret = hip_sqrt(hin, hres);
    bool h2ret = hip_sqrth2(hin, h2res);
    if (not (dret and fret and hret and h2ret))
    {
        std::cout << "vector add error!" << std::endl;
        return 1;
    }

    for (int i = 0; i < n; ++i)
    {
        float d = dres[i];
        float f = fres[i];
        float h = hres[i];
        float hH = h2res[i];

        if (fabs(d - f) > 0.01f or fabs(d - h) > 0.01f or fabs(f - h) > 0.01f or fabs(h - hH) > 0.01f)
        {
            std::cout << "d[" << i << "] = " << d << ", f[" << i << "] = " << f << ", h[" << i << "] = " << h << ", hH[" << i << "] = " << hH << std::endl;
        }
    }

    cout << "PASSED" << endl;

    return 0;
}

