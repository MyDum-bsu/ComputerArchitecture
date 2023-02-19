#ifndef LABS_VECTOR_H
#define LABS_VECTOR_H

#include <vector>

class Vector {
public:
    explicit Vector(int length);
    void multWithMPI(Vector, int mpi_size, int mpi_rank, int& result);
    int operator[](std::size_t) const;
    int* begin() { return vector_.data(); }
    int* end() { return vector_.data() + length_; }
    void generateVector();
private:
    int length_;
    std::vector<int> vector_;
};


#endif //LABS_VECTOR_H
