#ifndef __MATRIX_MUL_HPP__
#define __MATRIX_MUL_HPP__
#include <iostream>
#include <thread>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <cassert>

using namespace std;

template<class T>
class CMatrix {
public:
    CMatrix(bool row_major = true);
    CMatrix(size_t row, size_t column, bool row_major = true);
    CMatrix(const CMatrix<T> &matrix);

    ~CMatrix();

    void get_size(size_t &row, size_t &column) {
        row = m_row;
        column = m_column;
    }

    void resize(size_t row, size_t column) {
        if (m_row * m_column < row * column) {
            if (m_buffer != nullptr) {
                delete []m_buffer;
            }

            m_buffer = new T[row * column];
        }

        m_row = row;
        m_column = column;
    }

    T &get_elem(size_t i, size_t j) {
        if (m_row_major) {
            return m_buffer[i * m_column + j];
        } 
        else {
            return m_buffer[i + j * m_row];
        }
    }

    T &get_elem(size_t i, size_t j) const {
        if (m_row_major) {
            return m_buffer[i * m_column + j];
        } 
        else {
            return m_buffer[i + j * m_row];
        }
    }

    T *get_buffer() {
        return m_buffer;
    }

    void reset() {
        if (m_buffer != nullptr) {
            memset(m_buffer, 0, sizeof(T) * m_row * m_column);
        }
    }

    void set_row_major(bool row_major = true) {
        this->m_row_major = row_major;
    }

    bool is_row_major() {
        return m_row_major;
    }

    void print_partial(size_t row, size_t column);

    template<class T1>
    bool multiply(CMatrix<T> &matrix, CMatrix<T1> &res_matrix) {
        if (m_column != matrix.m_row) {
            return false;
        }

        if (res_matrix.m_buffer != nullptr) {
            delete []res_matrix.m_buffer;
        }

        res_matrix.m_row = m_row;
        res_matrix.m_column = matrix.m_column;
        assert(m_column == matrix.m_row);

        res_matrix.m_buffer = new T1[m_row * matrix.m_column];

        size_t i, j, k;
        for (i = 0; i < m_row; ++i) {
            for (j = 0; j < matrix.m_column; ++j) {
                T1 sum = 0;
                for (k = 0; k < m_column; k++) {
                    sum += static_cast<T1>(get_elem(i, k) * matrix.get_elem(k, j));
                }
                res_matrix.get_elem(i, j) = sum;
            }
        }

        return true;
    }

    template<class T1>
    bool multiply_optim(CMatrix<T> &matrix, CMatrix<T1> &res_matrix) {
        if (m_column != matrix.m_row) {
            return false;
        }

        if (res_matrix.m_buffer != nullptr) {
            delete []res_matrix.m_buffer;
        }

        res_matrix.m_row = m_row;
        res_matrix.m_column = matrix.m_column;
        assert(m_column == matrix.m_row);

        // if we convert the input matrix to column-major
        // format, the multiplication will be faster
        size_t i, j, k;
        T *a_tmp_buffer = nullptr;
        if (m_row_major) {
            a_tmp_buffer = m_buffer;
        }
        else {
            a_tmp_buffer = new T[m_row * m_column];
            for (i = 0; i < m_row; ++i) {
                for (j = 0; j < m_column; ++j) {
                    a_tmp_buffer[i * m_column + j] = m_buffer[i + j * m_row];
                }
            }
        }

        T *b_tmp_buffer = nullptr;
        if (matrix.is_row_major()) {
            b_tmp_buffer = new T[matrix.m_row * matrix.m_column];
            for (i = 0; i < matrix.m_row; ++i) {
                for (j = 0; j < matrix.m_column; ++j) {
                    b_tmp_buffer[i + j * matrix.m_row] = matrix.get_elem(i, j);
                }
            }
        }
        else {
            b_tmp_buffer = matrix.get_buffer();
        }

        res_matrix.m_buffer = new T1[m_row * matrix.m_column];
        for (i = 0; i < m_row; ++i) {
            for (j = 0; j < matrix.m_column; ++j) {
                T1 sum = 0;
                for (k = 0; k < m_column; k++) {
                    sum += a_tmp_buffer[i * m_column + k] * b_tmp_buffer[k + j * matrix.m_row];
                }
                res_matrix.get_elem(i, j) = sum;
            }
        }

        if (not m_row_major) {
            delete a_tmp_buffer;
        }
        if (matrix.is_row_major()) {
            delete b_tmp_buffer;
        }

        return true;
    }

    template<class T1>
    bool multiply_parallel(const CMatrix<T> &matrix, size_t thread_num, CMatrix<T1> &res_matrix) {
        if (m_column != matrix.m_row) {
            return false;
        }

        if (res_matrix.m_buffer != nullptr) {
            delete []res_matrix.m_buffer;
        }

        res_matrix.m_row = m_row;
        res_matrix.m_column = matrix.m_column;
        res_matrix.m_buffer = new T1[m_row * matrix.m_column];

        // Create multiple threads for the computation
        size_t thread_idx;
        vector<thread> vec_tid(thread_num);
        for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
            vec_tid[thread_idx] = thread(&CMatrix<T>::multiply_thread<T1>, this, matrix, thread_idx, thread_num, std::ref(res_matrix));
        }

        for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
            vec_tid[thread_idx].join();
        }

        return true;
    }
    
    bool operator == (const CMatrix<T> &matrix);
    bool operator != (const CMatrix<T> &matrix);

private:
    // thread function to calculate partial matrix
    template<class T1>
    void multiply_thread(const CMatrix<T> &matrix, size_t thread_idx, size_t thread_num, CMatrix<T1> &res_matrix) {
        size_t rows_per_thread = m_row / thread_num;
        size_t start_row = thread_idx * rows_per_thread;
        size_t end_row = start_row + rows_per_thread;
        size_t i, j, k;
        for (i = start_row; i < end_row; ++i) {
            for (j = 0; j < res_matrix.m_column; ++j) {
                T1 sum = 0;
                for (k = 0; k < m_column; ++k) {
                    sum += get_elem(i, k) * matrix.get_elem(k, j);
                }
                res_matrix.get_elem(i, j) = sum;
            }
        }

        return;
    }

   void init_();

public:
    size_t m_row, m_column;
    T *m_buffer;
    bool m_row_major;
};

template<class T>
CMatrix<T>::CMatrix(bool row_major) : m_row(0), m_column(0), m_row_major(true) {
    m_buffer = nullptr;
}

template<class T>
CMatrix<T>::CMatrix(size_t row, size_t column, bool row_major) : m_row(row), m_column(column), m_row_major(row_major) {
    init_();
}

template<class T>
CMatrix<T>::CMatrix(const CMatrix<T> &matrix) {
    m_column = matrix.m_column;
    m_row = matrix.m_row;
    m_row_major = matrix.m_row_major;
    m_buffer = new T[m_row * m_column];
    memcpy(m_buffer, matrix.m_buffer, sizeof(T) * m_row * m_column);

    return;
}

template<class T>
CMatrix<T>::~CMatrix() {
    if (m_buffer) {
        delete []m_buffer;
    }
    m_buffer = nullptr;

    m_row = m_column = 1;
}

template<class T>
void CMatrix<T>::print_partial(size_t row, size_t column) {
    size_t i, j;
    cout << "Matrix is:" << endl;
    for (i = 0; i < row; ++i) {
        for (j = 0; j < column; ++j) {
            cout << this->get_elem(i, j) << " ";
        }
        cout << endl;
    }
    cout << endl;
}


// template<class T>
// bool CMatrix<T>::multiply_optim(CMatrix<T> &matrix, CMatrix<T> &res_matrix) {
// }

// template<class T>
// bool CMatrix<T>::multiply_parallel(const CMatrix<T> &matrix, size_t thread_num, CMatrix<T> &res_matrix) {
// }

// template<class T>
// void CMatrix<T>::multiply_thread(const CMatrix<T> &matrix, size_t thread_idx, size_t thread_num, CMatrix<T> &res_matrix) {
// }

template<class T>
void CMatrix<T>::init_() {
    srand(time(nullptr));
    m_buffer = new T[m_row * m_column];
    if (m_row_major) {
        for (size_t i = 0; i < m_row; ++i) {
            for (size_t j = 0; j < m_column; ++j) {
                size_t loc = i * m_column + j;
                int r = rand() % 100;
                m_buffer[loc] = static_cast<T>(r / 10.0f);
            }
        }
    }
    else {
        for (size_t i = 0; i < m_row; ++i) {
            for (size_t j = 0; j < m_column; ++j) {
                size_t loc = j * m_row + i;
                int r = rand() % 100;
                m_buffer[loc] = static_cast<T>(r / 10.0f);
            }
        }
    }

    return;
}

template<class T>
bool CMatrix<T>::operator == (const CMatrix<T> &matrix) {
    if (m_row_major != matrix.m_row_major) {
        return false;
    } 
    if (m_row != matrix.m_row) {
        return false;
    }
    if (m_column != matrix.m_column) {
        return false;
    }

    T atol = 1e-2f;
    T rtol = 1e-2f;
    size_t i, len = m_row * m_column;
    for (i = 0; i < len; ++i) {
        if (std::fabs(m_buffer[i] - matrix.m_buffer[i]) > atol + rtol * m_buffer[i]) {
            cout << "m1[" << i << "] = " << m_buffer[i] << ", ";
            cout << "m2[" << i << "] = " << matrix.m_buffer[i] << endl;
            return false;
        }
    }

    return true;
}

// template<>
// bool std::enable_if_t<std::is_integral_v<T>> CMatrix<int32_t>::operator == (const CMatrix<int32_t> &matrix) {
//     if (m_row_major != matrix.m_row_major) {
//         return false;
//     } 
//     if (m_row != matrix.m_row) {
//         return false;
//     }
//     if (m_column != matrix.m_column) {
//         return false;
//     }

//     size_t i, len = m_row * m_column;
//     for (i = 0; i < len; ++i) {
//         if (m_buffer[i] != matrix.m_buffer[i]) return false;
//     }

//     return true;
// }


template<class T>
bool CMatrix<T>::operator != (const CMatrix<T> &matrix) {
    return not(*this == matrix);
}

#endif

