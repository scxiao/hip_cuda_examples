#ifndef __MATRIX_MUL_HPP__
#define __MATRIX_MUL_HPP__
#include <iostream>
#include <thread>
#include <vector>
#include <cstring>
#include <algorithm>

using namespace std;

template<class T>
class CMatrix {
public:
    CMatrix();
    CMatrix(size_t row, size_t column);
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
        return m_buffer[i * m_column + j];
    }

    T *get_buffer() {
        return m_buffer;
    }

    void reset() {
        if (m_buffer != nullptr) {
            memset(m_buffer, 0, sizeof(T) * m_row * m_column);
        }
    }

    void print_partial(size_t row, size_t column);

    bool multiply(CMatrix<T> &matrix, CMatrix<T> &res_matrix);
    bool multiply_optim(CMatrix<T> &matrix, CMatrix<T> &res_matrix);
    bool multiply_parallel(CMatrix<T> &matrix, size_t thread_num, CMatrix<T> &res_matrix);
    
    bool operator == (const CMatrix<T> &matrix);
    bool operator != (const CMatrix<T> &matrix);

private:
     // thread function to calculate partial matrix
    void multiply_thread(T *buffer, size_t thread_idx, size_t thread_num, CMatrix<T> &res_matrix);

   void init_();

private:
    size_t m_row, m_column;
    T *m_buffer;
};

template<class T>
CMatrix<T>::CMatrix() : m_row(0), m_column(0) {
    m_buffer = nullptr;
}

template<class T>
CMatrix<T>::CMatrix(size_t row, size_t column) : m_row(row), m_column(column) {
    init_();
}

template<class T>
CMatrix<T>::CMatrix(const CMatrix<T> &matrix) {
    m_column = matrix.m_column;
    m_row = matrix.m_row;
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
            size_t loc = i * m_column + j;
            cout << m_buffer[loc] << " ";
        }
        cout << endl;
    }
    cout << endl;
}


template<class T>
bool CMatrix<T>::multiply(CMatrix<T> &matrix, CMatrix<T> &res_matrix) {
    if (m_column != matrix.m_row) {
        return false;
    }

    if (res_matrix.m_buffer != nullptr) {
        delete []res_matrix.m_buffer;
    }

    res_matrix.m_row = m_row;
    res_matrix.m_column = matrix.m_column;

    res_matrix.m_buffer = new T[m_row * matrix.m_column];
    size_t i, j, k;
    for (i = 0; i < m_row; ++i) {
        for (j = 0; j < matrix.m_column; ++j) {
            T sum = 0;
            for (k = 0; k < m_column; k++) {
                sum += get_elem(i, k) * matrix.get_elem(k, j);
            }
            res_matrix.get_elem(i, j) = sum;
        }
    }

    return true;
}

template<class T>
bool CMatrix<T>::multiply_optim(CMatrix<T> &matrix, CMatrix<T> &res_matrix) {
    if (m_column != matrix.m_row) {
        return false;
    }

    if (res_matrix.m_buffer != nullptr) {
        delete []res_matrix.m_buffer;
    }

    res_matrix.m_row = m_row;
    res_matrix.m_column = matrix.m_column;

    // if we convert the input matrix to column-major
    // format, the multiplication will be faster
    size_t i, j, k;
    T *tmp_buffer = new T[matrix.m_row * matrix.m_column];
    for (i = 0; i < matrix.m_row; ++i) {
        for (j = 0; j < matrix.m_column; ++j) {
            tmp_buffer[i + j * matrix.m_row] = matrix.get_elem(i, j);
        }
    }

    res_matrix.m_buffer = new T[m_row * matrix.m_column];
    for (i = 0; i < m_row; ++i) {
        for (j = 0; j < matrix.m_column; ++j) {
            T sum = 0;
            for (k = 0; k < m_column; k++) {
                sum += get_elem(i, k) * tmp_buffer[k + j * matrix.m_row];
            }
            res_matrix.get_elem(i, j) = sum;
        }
    }

    delete[] tmp_buffer;

    return true;
}

template<class T>
bool CMatrix<T>::multiply_parallel(CMatrix<T> &matrix, size_t thread_num, CMatrix<T> &res_matrix) {
    if (m_column != matrix.m_row) {
        return false;
    }

    if (res_matrix.m_buffer != nullptr) {
        delete []res_matrix.m_buffer;
    }

    res_matrix.m_row = m_row;
    res_matrix.m_column = matrix.m_column;
    res_matrix.m_buffer = new T[m_row * matrix.m_column];

    // if we convert the input matrix to column-major
    // format, the multiplication will be faster
    size_t i, j;
    T *tmp_buffer = new T[matrix.m_row * matrix.m_column];
    for (i = 0; i < matrix.m_row; ++i) {
        for (j = 0; j < matrix.m_column; ++j) {
            tmp_buffer[i + j * matrix.m_row] = matrix.get_elem(i, j);
        }
    }

    // Create multiple threads for the computation
    size_t thread_idx;
    vector<thread> vec_tid(thread_num);
    for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        vec_tid[thread_idx] = thread(&CMatrix<T>::multiply_thread, this, tmp_buffer, thread_idx, thread_num, std::ref(res_matrix));
    }

    for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        vec_tid[thread_idx].join();
    }

    delete[] tmp_buffer;

    return true;
}

template<class T>
void CMatrix<T>::multiply_thread(T *buffer, size_t thread_idx, size_t thread_num, CMatrix<T> &res_matrix) {
    size_t rows_per_thread = m_row / thread_num;
    size_t start_row = thread_idx * rows_per_thread;
    size_t end_row = start_row + rows_per_thread;
    //cout << "threadidex = " << thread_idx << ",start_row = " << start_row << ", end_row = " << end_row << std::endl;
    size_t i, j, k;
    for (i = start_row; i < end_row; ++i) {
        for (j = 0; j < res_matrix.m_column; ++j) {
            T sum = 0;
            for (k = 0; k < m_column; ++k) {
                sum += get_elem(i, k) * buffer[k + j * m_column];
            }
            res_matrix.get_elem(i, j) = sum;
        }
    }

    return;
}

template<class T>
void CMatrix<T>::init_() {
    srand(time(nullptr));
    m_buffer = new T[m_row * m_column];
    for (size_t i = 0; i < m_row; ++i) {
        for (size_t j = 0; j < m_column; ++j) {
            size_t loc = i * m_column + j;
            int r = rand() % 10;
            m_buffer[loc] = static_cast<T>(r);
        }
    }

    return;
}

template<class T>
bool CMatrix<T>::operator == (const CMatrix<T> &matrix) {
    if (m_row != matrix.m_row) {
        return false;
    }

    if (m_column != matrix.m_column) {
        return false;
    }

    size_t i, len = m_row * m_column;
    for (i = 0; i < len; ++i) {
        if (m_buffer[i] != matrix.m_buffer[i]) {
            cout << "m1[" << i << "] = " << m_buffer[i] << ", ";
            cout << "m2[" << i << "] = " << matrix.m_buffer[i] << endl;
            return false;
        }
    }

    return true;
}

template<class T>
bool CMatrix<T>::operator != (const CMatrix<T> &matrix) {
    return not(*this == matrix);
}

#endif

