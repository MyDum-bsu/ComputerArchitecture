#include <iostream>
#include <mpi.h>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

int main(int argc, char** argv) {
    int L, rank, size, quotient, remainder;
    double local_sum = 0, global_sum;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc != 2) {
        if (rank == 0) {
            cout << "Usage: mpirun -n <num_processes> ./executable <vector_length>" << endl;
        }
        MPI_Finalize();
        return 1;
    }
    L = atoi(argv[1]);
    vector<double> vec1(L);
    vector<double> vec2(L);
    srand(time(NULL));
    for (int i = 0; i < L; i++) {
        vec1[i] = rand() % 2 - 1;
        vec2[i] = rand() % 2 - 1;
    }
    quotient = L / size;
    remainder = L % size;
    vector<double> local_vec1(quotient + (rank < remainder ? 1 : 0));
    vector<double> local_vec2(quotient + (rank < remainder ? 1 : 0));
    MPI_Scatter(vec1.data(), quotient + (rank < remainder ? 1 : 0), MPI_DOUBLE,
                local_vec1.data(), quotient + (rank < remainder ? 1 : 0), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Scatter(vec2.data(), quotient + (rank < remainder ? 1 : 0), MPI_DOUBLE,
                local_vec2.data(), quotient + (rank < remainder ? 1 : 0), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    for (int i = 0; i < local_vec1.size(); i++) {
        local_sum += local_vec1[i] * local_vec2[i];
    }
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        cout << "Vector 1: ";
        for (auto v : vec1) {
            cout << v << " ";
        }
        cout << endl << "Vector 2: ";
        for (auto v : vec2) {
            cout << v << " ";
        }
        cout << endl << "Scalar product: " << global_sum << endl;
    }
    MPI_Finalize();
    return 0;
}