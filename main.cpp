#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <string.h>
#include <CL/cl.hpp>
#include <kernel_code.h>

using namespace std;
using namespace chrono;



// Funkcja do odczytywania błędów kompilacji programu OpenCL
void checkOpenCLBuildError(cl::Program& program, const cl::Device& device) {
    cl_build_status buildStatus;
    program.getBuildInfo(device, CL_PROGRAM_BUILD_STATUS, &buildStatus);
    
    if (buildStatus != CL_BUILD_SUCCESS) {
        string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        cout << "OpenCL program build error:" << endl;
        cout << buildLog << endl;
    }
}

double determinantCL(double** mat, int n, cl::Context& context, cl::CommandQueue& queue, cl::Kernel& kernel) {
     double* matrix = new double[n * n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix[i * n + j] = mat[i][j];
            }
        }

    size_t matrixSize = n * n * sizeof(double);
    size_t resultSize = n * sizeof(double);

    cl::Buffer matrixBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, matrix);
    cl::Buffer resultBuffer(context, CL_MEM_WRITE_ONLY, resultSize);
                     
    kernel.setArg(0, matrixBuffer);
    kernel.setArg(1, resultBuffer);
    kernel.setArg(2, n);
    
    cl::NDRange global(n);
    cl::NDRange local(1, 1);
            
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);

    double* result = new double[n];
    queue.enqueueReadBuffer(resultBuffer, CL_TRUE, 0, resultSize, result);
    
    double final_result = 1.0;
    for (int i = 0; i < n; ++i) {
        final_result *= result[i];
    }
    
    delete[] result;
    
    return final_result;
}

double determinant(double** mat, int n) {
    double det = 1.0;
    
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double ratio = mat[j][i] / mat[i][i];
            for (int k = 0; k < n; ++k) {
                mat[j][k] -= ratio * mat[i][k];
            }
        }
        det *= mat[i][i];
    }
    
    return det;
}

double determinantMP(double** mat, int n) {
    double det = 1.0;
    
    #pragma omp parallel for reduction(*:det)
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double ratio = mat[j][i] / mat[i][i];
            for (int k = 0; k < n; ++k) {
                mat[j][k] -= ratio * mat[i][k];
            }
        }
        det *= mat[i][i];
    }
    
    return det;
}

void generateMatrix(double** matrix, int size) {
    int min_value = -100;
    int max_value = 100;
    
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(min_value, max_value);
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = dis(gen);
        }
    }
}

int main() {
    ofstream file("wyniki.csv");
    if (!file.is_open()) {
        cerr << "Error: Cannot open file for writing" << endl;
        return 1;
    }
    
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
    cl::Context context(CL_DEVICE_TYPE_GPU, properties);

    vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    cl::CommandQueue queue(context, devices[0]);

    cl::Program::Sources sources;
    sources.push_back({kernel_code.c_str(), kernel_code.length()});
    cl::Program program(context, sources);
    
    if (program.build({devices[0]}) != CL_SUCCESS) {
        checkOpenCLBuildError(program, devices[0]);
        return 1;
    }

    cl::Kernel kernel(program, "determinant");

    
    file << "Size;CPU;CPU_OpenMP;GPU_OpenCL" << endl;
    
    for (int size = 100; size <= 150; size += 10) {
        double** matrix = new double*[size];
        for (int i = 0; i < size; ++i) {
            matrix[i] = new double[size];
        }
        
        generateMatrix(matrix, size);
        
        auto start = high_resolution_clock::now();
        determinant(matrix, size);
        auto end = high_resolution_clock::now();
        double time_cpu = duration_cast<duration<double>>(end - start).count();
        
        start = high_resolution_clock::now();
        determinantMP(matrix, size);
        end = high_resolution_clock::now();
        double time_cpu_mp = duration_cast<duration<double>>(end - start).count();
        
        start = high_resolution_clock::now();
        determinantCL(matrix, size, context, queue, kernel);
        end = high_resolution_clock::now();
        double time_gpu = duration_cast<duration<double>>(end - start).count();
        
        file << size << ";" << time_cpu << ";" << time_cpu_mp << ";" << time_gpu << endl;
        
        for (int i = 0; i < size; ++i) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }
    
    file.close();
    
    return 0;
}
