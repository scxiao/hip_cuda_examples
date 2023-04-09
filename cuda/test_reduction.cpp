#include <iostream>
#include <cstdlib>
#include "cu_reduction.hpp"
#include <init_vec.hpp>
#include <utilities.hpp>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " n (element num = 2^n)" << endl;
        return 0;
    }

    int sn = atoi(argv[1]);
    size_t n = (1 << sn);

    cout << "Element num: 2^" << sn << " = " << n << endl << std::endl;
    srand(time(nullptr));
    std::vector<float> in_vec;

    // init vector
    init_vec(in_vec, n);
    std::vector<half> in_vech(in_vec.begin(), in_vec.end());

    std::cout << "/////////////////////   FP32 /////////////////////////" << std::endl << std::endl;

    std::vector<float> out_vec0;
    reduction0(in_vec, out_vec0);
    std::cout << std::endl;

    std::vector<float> out_vec1;
    reduction1(in_vec, out_vec1);
    bool ret = compare(out_vec0, out_vec1, 0.01f);
    cout << "FP32 Reduction1 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<float> out_vec2;
    reduction2(in_vec, out_vec2);
    ret = compare(out_vec1, out_vec2, 0.01f);
    cout << "FP32 Reduction2 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<float> out_vec3;
    reduction3(in_vec, out_vec3);
    ret = compare(out_vec2, out_vec3, 0.01f);
    cout << "FP32 Reduction3 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<float> out_vec4;
    reduction4(in_vec, out_vec4);
    ret = compare(out_vec3, out_vec4, 0.01f);
    cout << "FP32 Reduction4 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<float> out_vec5;
    reduction5(in_vec, out_vec5);
    ret = compare(out_vec3, out_vec5, 0.01f);
    cout << "FP32 Reduction5 " << (ret ? "PASSED" : "FAILED") << endl << endl;


    std::cout << "/////////////////////   FP16 /////////////////////////" << std::endl << std::endl;

    std::vector<half> out_vec0h;
    reduction0(in_vech, out_vec0h);
    std::cout << std::endl;

    std::vector<half> out_vec1h;
    reduction1(in_vech, out_vec1h);
    ret = compare(out_vec0h, out_vec1h, 0.01f);
    cout << "FP16 Reduction1 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<half> out_vec2h;
    reduction2(in_vech, out_vec2h);
    ret = compare(out_vec1h, out_vec2h, 0.01f);
    cout << "FP16 Reduction2 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<half> out_vec3h;
    reduction3(in_vech, out_vec3h);
    ret = compare(out_vec2h, out_vec3h, 0.01f);
    cout << "FP16 Reduction3 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<half> out_vec4h;
    reduction4(in_vech, out_vec4h);
    ret = compare(out_vec3h, out_vec4h, 0.01f);
    cout << "FP16 Reduction4 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    std::vector<half> out_vec5h;
    reduction5(in_vech, out_vec5h);
    ret = compare(out_vec3h, out_vec5h, 0.01f);
    cout << "FP16 Reduction5 " << (ret ? "PASSED" : "FAILED") << endl << endl;

    return 0;
}

