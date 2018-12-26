#include <mpi.h>
#include <iostream>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
using namespace std;
int ARRAY_SIZE = 200;
int ITERATION_NUMBER = 500000;

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    // Find out rank, size
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    string infile_name = argv[1];
    string outfile_name = argv[2];
    double beta = atof(argv[3]);
    double pi = atof(argv[4]);
    double gama = 0.5 * log((1.0-pi) / pi);
    int num_of_slaves = world_size-1;  //number of slaves
    int slave_row_num = ARRAY_SIZE / num_of_slaves; //array is distributed among slaves.
    /**
     * if world_rank is 0, that means master processor will work,
     * otherwise, slaves will be proceeded.
     */
    if (world_rank == 0){
        //read matris from the file named infile_name.
        ifstream infile (infile_name);
        //X represents master processor's array.
        int X[ARRAY_SIZE][ARRAY_SIZE] ;
        int number = 0;
        if (infile.is_open())
        {
            for (int i = 0; i < ARRAY_SIZE; ++i)
                for (int j = 0; j < ARRAY_SIZE; ++j)
                    infile >> X[i][j];
            infile.close();
        }
        else cout << "Unable to open the input file";
        //holds all arrays hold by slaves.
        int subarr[num_of_slaves][slave_row_num][ARRAY_SIZE];
        //sends equal parts of the master array to slaves.
        for (int k = 1; k < world_size; k++)
            MPI_Send(X[slave_row_num*(k-1)], slave_row_num*ARRAY_SIZE, MPI_INT, k, 0, MPI_COMM_WORLD);
        //receives proceeded arrays from slaves.
        for (int k = 1; k < world_size; k++)
            MPI_Recv(subarr[k-1][0], slave_row_num*ARRAY_SIZE, MPI_INT, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        ofstream outfile;
        outfile.open(outfile_name);
        //write slave arrays to an output file
        for (int l = 0; l < num_of_slaves; ++l) {
            for (int i = 0; i < slave_row_num; ++i) {
                for (int j = 0; j < ARRAY_SIZE; ++j) {
                    outfile << subarr[l][i][j] << " ";
                }
                outfile << "\n";
            }
        }
        outfile.close();
    }else{
        //holds the original of the slave array.
        int arrayForSlave[slave_row_num][ARRAY_SIZE];
        MPI_Recv(arrayForSlave[0], slave_row_num*ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //changes will be made in temp array.
        int temp[slave_row_num][ARRAY_SIZE];
        for (int i = 0; i < slave_row_num ; ++i) {
            for (int j = 0; j < ARRAY_SIZE; ++j) {
                temp[i][j] = arrayForSlave[i][j];
            }
        }
        int calculationArr[slave_row_num+2][ARRAY_SIZE+2];
        srand(time(0) + world_rank);  // Initialize random number generator. We want different seeds for different ranks.
        for (int k = 0; k < ITERATION_NUMBER; ++k) {
            int received_top[ARRAY_SIZE];
            int received_bottom[ARRAY_SIZE];
            for (int i = 0; i < ARRAY_SIZE; i++) {
               received_top[i] = 0;
               received_bottom[i] = 0;
            }
            int rand_row = (rand() % (slave_row_num));
            int rand_col = (rand() % 200);

             //top and bottom processors will work different that the ones at middle.
            if (world_rank == 1){
              //if there is just one slave, don't send and receive from anywhere.
              if(world_size != 2){
                MPI_Send(temp[slave_row_num-1], ARRAY_SIZE, MPI_INT, 2, 0, MPI_COMM_WORLD);
                MPI_Recv(received_bottom, ARRAY_SIZE, MPI_INT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
              }
            }else if(world_rank == world_size -1){
                MPI_Send(temp[0], ARRAY_SIZE, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD);
                MPI_Recv(received_top, ARRAY_SIZE, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }else{
                MPI_Send(temp[slave_row_num-1], ARRAY_SIZE, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD);
                MPI_Send(temp[0], ARRAY_SIZE, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(received_bottom, ARRAY_SIZE, MPI_INT, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(received_top, ARRAY_SIZE, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
             //calculate neighbours
            for (int i = 0; i < slave_row_num+2; i++) {
              for (int j = 0; j < ARRAY_SIZE+2; j++) {
                calculationArr[i][j] = 0;
              }
              
            }
            for (int j = 1; j < ARRAY_SIZE+1; j++) {
                calculationArr[0][j] = received_top[j-1];
            }

            for (int i = 1; i < slave_row_num+1; i++) {
              for (int j = 1; j < ARRAY_SIZE+1; j++) {
                calculationArr[i][j] = temp[i-1][j-1];
              }
            }
            for (int j = 1; j < ARRAY_SIZE+1; j++) {
                calculationArr[slave_row_num+1][j] = received_bottom[j-1];
            }
            int sum = calculationArr[rand_row][rand_col] + calculationArr[rand_row][rand_col+1] + calculationArr[rand_row+1][rand_col] +
              calculationArr[rand_row+2][rand_col+2] + calculationArr[rand_row+2][rand_col+1] +
              calculationArr[rand_row+1][rand_col+2] + calculationArr[rand_row][rand_col+2] + calculationArr[rand_row+2][rand_col];
            double acceptance_prob = (-2.0)*gama*(double)arrayForSlave[rand_row][rand_col]*(double)temp[rand_row][rand_col]
                                -(2.0)*beta*(double)temp[rand_row][rand_col]*(double)sum;
            double rand_data = ((double) rand() / (RAND_MAX));
            if (rand_data < exp(acceptance_prob))
              temp[rand_row][rand_col] = -1*temp[rand_row][rand_col]; //Update the image
        }
        MPI_Send(temp[0], slave_row_num*ARRAY_SIZE, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
}
