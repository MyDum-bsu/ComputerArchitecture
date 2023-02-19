#include <iostream>
#include <mpi.h>
#include "first/Vector.h"

void initMPI(int argc, char **argv, int &rank,  int &size);

void multVectors(int L, int rank, int size);

int main(int argc, char **argv) {
    int rank, size;
    initMPI(argc, argv, rank, size);
    multVectors(atoi(argv[1]), rank, size);
    MPI_Finalize();
    return 0;
}

void multVectors(int L, int rank, int size) {
    Vector vec1(L);
    vec1.generateVector();
    Vector vec2(L);
    vec2.generateVector();
    int res;
    vec1.multWithMPI(vec2, size, rank, res);
    if (rank == 0) {
        std::cout << "Vector's size: " << L << "\n";
        std::cout << "Vector 1: ";
        for (auto v: vec1) {
            std::cout << v << " ";
        }
        std::cout << "\nVector 2: ";
        for (auto v: vec2) {
            std::cout << v << " ";
        }
        std::cout << "\nScalar product: " << res << "\n";
    }
}

void initMPI(int argc, char **argv, int &rank, int &size) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 2) {
        if (rank == 0) {
            throw std::runtime_error("Usage: mpirun -n <num_processes> ./executable <vector_length>\\n\"");
        }
        MPI_Finalize();
    }
}