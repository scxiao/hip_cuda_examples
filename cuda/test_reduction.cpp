#include <iostream>
#include <cstdlib>
#include "cu_reduction.hpp"
#include <init_vec.hpp>
#include <utilities.hpp>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " n" << endl;
        return 0;
    }

    int sn = atoi(argv[1]);
    size_t n = (1 << sn);

    cout << "Element num: " << n << endl;
    srand(time(nullptr));
    std::vector<float> in_vec;

    // init vector
    init_vec(in_vec, n);

    std::vector<float> out_vec0;
    reduction0(in_vec, out_vec0);
    std::cout << std::endl;

    std::vector<float> out_vec1;
    reduction1(in_vec, out_vec1);
    bool ret = compare(out_vec0, out_vec1);
    cout << "Reduction1 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<float> out_vec2;
    reduction2(in_vec, out_vec2);
    ret = compare(out_vec1, out_vec2);
    cout << "Reduction2 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<float> out_vec3;
    reduction3(in_vec, out_vec3);
    ret = compare(out_vec2, out_vec3);
    cout << "Reduction3 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<float> out_vec4;
    reduction4(in_vec, out_vec4);
    ret = compare(out_vec3, out_vec4);
    cout << "Reduction4 " << (ret ? "PASSED" : "FAILED") << endl << endl;


    return 0;
}

