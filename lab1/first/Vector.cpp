#include "Vector.h"
#include <random>
#include <iostream>
#include <mpi.h>

Vector::Vector(int length) : length_(length) {
    vector_ = std::vector<int>(length_);
}

void Vector::generateVector() {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(-1, 0);
    vector_ = std::vector<int>(length_);
    for (int i = 0; i < length_; i++) {
        vector_[i] = dist(mt);
    }
}

void Vector::multWithMPI(Vector v, int mpi_size, int mpi_rank, int &result) {
    if (v.length_ != length_) {
        throw std::runtime_error("different length");
    }

    int quotient = length_ / mpi_size;
    int remainder = length_ % mpi_size;
    if (remainder != 0) {
        std::vector<int> vec(remainder, 0);
        vector_.insert(vector_.end(), vec.begin(), vec.end());
        v.vector_.insert(v.vector_.end(), vec.begin(), vec.end());
    }
    int sendcount = quotient + (mpi_rank < remainder);
    Vector loc_vec1(sendcount);
    Vector loc_vec2(sendcount);
    MPI_Scatter(vector_.data(), sendcount, MPI_INT,
                loc_vec1.vector_.data(), sendcount, MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Scatter(v.vector_.data(), sendcount, MPI_INT,
                loc_vec2.vector_.data(), sendcount, MPI_INT,
                0, MPI_COMM_WORLD);
    int loc_sum = 0;
    for (int i = 0; i < loc_vec1.length_; i++) {
        loc_sum += loc_vec1[i] * loc_vec2[i];
    }
    MPI_Reduce(&loc_sum, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
}

int Vector::operator[](size_t index) const {
    if (index >= vector_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return vector_[index];
}