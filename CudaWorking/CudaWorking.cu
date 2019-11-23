#include <iostream>
#include <fstream>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define LINE_LENGTH 20 // for now, fix it;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

struct returnStructure {
    int index;
    int lines;
    int isSet;
};

__device__ int devStrcmp(char* comp, char* comp_two, int lsize, int rsize) {

    if (lsize != rsize) {
        return 0;
    }

    for (int a = 0; a < lsize; a++) {
        if (comp[a] != comp_two[a]) {
            return 0;
        }
    }
    return 1;
}

__global__ void compareString(char* ptr, char* compString, int* compare_length, returnStructure *rs, int* limit) {
    char arr_tmp[3];
    int block_id_index = (blockDim.x * blockIdx.x + threadIdx.x);
    //printf("st_Array: %d\n", block_id_index);
    if (block_id_index >= *limit) return;
    int st_array = block_id_index * LINE_LENGTH; // To use thread and block more efficiently, use x/y + thread idx
    int length;

    // The Process goes on - Making Array
    for (int i = 0; i < LINE_LENGTH; i++, st_array++) {
        if (ptr[st_array] != 0) {
            arr_tmp[i] = ptr[st_array];
        } else {
            arr_tmp[i] = 0;
            length = i;
            break;
        }
    }
    if (!devStrcmp(arr_tmp, compString, length, *compare_length)) {
        //printf("Not Same!\n");
    } else {
        rs[block_id_index].index = st_array;
        rs[block_id_index].lines = block_id_index;
        rs[block_id_index].isSet = 1;
    }
}

__global__ void printArray(char* ptr, int* ctr) {
    char arr_tmp[3];
    int st_array = (blockDim.x * blockIdx.x + threadIdx.x) * LINE_LENGTH; // To use thread and block more efficiently, use x/y + thread idx
    //printf("st_Array: %d\n", st_array);

    // The Process goes on - Making Array
    for (int i = 0; i < LINE_LENGTH; i++, st_array++) {
        if (ptr[st_array] != 0) {
            arr_tmp[i] = ptr[st_array];
        } else {
            arr_tmp[i] = 0;
            break;
        }
    }
    //printf("The Index: %d and array index is: %d\n", st_array, (blockDim.x * blockIdx.x + threadIdx.x));
    ctr[(blockDim.x * blockIdx.x + threadIdx.x)] = 10;
}

int main(void) {

    // Host Constant variables.
    const int LN_LIMIT_PER_BLOCK = 1024;

    // Host Variable for Cuda Iteration.(Block Count)
    int BLOCK_CTR;

    // Host-Related variables
    char* dev_Array;
    char** read_input = nullptr;
    int line_ctr;
    std::string compStringVar = "90";
    int stringValLength = compStringVar.length();
    int* returnArray;
    struct returnStructure* rs;

    // Device-Related variables
    char* real_dev;
    char* dev_com_string;
    int* compStringLength;
    int* test_array;
    int* limit_exceed_chk;
    struct returnStructure* dev_rs;

    // The Host Code
    std::string tmp;
    std::ifstream fst("C:\\Users\\KangDroid\\Desktop\\test.txt");
    std::ifstream if_ctr("C:\\Users\\KangDroid\\Desktop\\test.txt");
    line_ctr = std::count(std::istreambuf_iterator<char>(if_ctr),
        std::istreambuf_iterator<char>(), '\n');
    printf("Line count: %d\n", line_ctr);

    // Calculate how many iterations we need to act.
    BLOCK_CTR = line_ctr / LN_LIMIT_PER_BLOCK;
    if ((line_ctr % LN_LIMIT_PER_BLOCK) != 0) {
        BLOCK_CTR++;
    } // And line_ctr is the limit.

    // No init the thing.
    returnArray = new int[line_ctr];
    rs = new returnStructure[line_ctr];

    read_input = new char* [line_ctr];
    for (int i = 0; i < line_ctr; i++) {
        fst >> tmp;
        read_input[i] = new char[LINE_LENGTH]; // The Null value
        strcpy(read_input[i], tmp.c_str());
        //printf("%s\n", read_input[i]);
    }

    int ctr_devarr = line_ctr * LINE_LENGTH;
    dev_Array = new char[ctr_devarr];
    int ctr_arr_aux = 0;

    // Copy it to First Dim Array;
    for (int i = 0; i < line_ctr; i++) {
        for (int a = 0; a < LINE_LENGTH; a++) {
            dev_Array[ctr_arr_aux] = read_input[i][a];
            ctr_arr_aux++;
        }
    }

    // Cuda Kernel Call
    gpuErrchk(cudaMalloc((void**)&real_dev, sizeof(char) * ctr_devarr));
    gpuErrchk(cudaMalloc((void**)&dev_com_string, sizeof(char) * compStringVar.length()));
    gpuErrchk(cudaMalloc((void**)&compStringLength, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&test_array, sizeof(int) * line_ctr));
    gpuErrchk(cudaMalloc((void**)&dev_rs, sizeof(struct returnStructure) * line_ctr));
    gpuErrchk(cudaMalloc((void**)&limit_exceed_chk, sizeof(int)));
    gpuErrchk(cudaMemcpy(limit_exceed_chk, &line_ctr, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_com_string, compStringVar.c_str(), sizeof(char) * compStringVar.length(), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(compStringLength, &stringValLength, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(real_dev, dev_Array, sizeof(char) * ctr_devarr, cudaMemcpyHostToDevice));

    printf("Block: %d\n", BLOCK_CTR);
    compareString << <BLOCK_CTR, LN_LIMIT_PER_BLOCK >> > (real_dev, dev_com_string, compStringLength, dev_rs, limit_exceed_chk);
    //printArray << <20, 512 >> > (real_dev, test_array);
    gpuErrchk(cudaMemcpy(returnArray, test_array, sizeof(int) * line_ctr, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(rs, dev_rs, sizeof(struct returnStructure) * line_ctr, cudaMemcpyDeviceToHost));
    for (int i = 0; i < line_ctr; i++) {
        if (rs[i].isSet == 1) {
            printf("Lines: %d\n", rs[i].lines + 1);
        }
    }

    // Remove Dynamically allocated memories.
    cudaFree(dev_Array);
    cudaFree(dev_com_string);
    cudaFree(compStringLength);
    cudaFree(test_array);
    cudaFree(dev_rs);
    cudaFree(limit_exceed_chk);

    for (int i = 0; i < line_ctr; i++) {
        delete[] read_input[i];
    }
    delete[] read_input;
    delete[] dev_Array;
    delete[] returnArray;
    delete[] rs;
    return 0;
}