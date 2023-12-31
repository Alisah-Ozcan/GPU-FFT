#include <cstdlib>  // For atoi or atof functions
#include <iomanip>
#include <iostream>
#include <random>

#include "../src/fft.cuh"
#include "../src/fft_cpu.cuh"

using namespace std;

int q;
int logn;
int batch;
int n;

int main(int argc, char* argv[])
{
    if (argc == 0)
    {
        q = 7;
        logn = 11;
        batch = 1;
        n = 1 << logn;
    }
    else
    {
        q = 7;
        logn = atoi(argv[1]);
        batch = atoi(argv[2]);
        n = 1 << logn;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long minNumber = 0;
    unsigned long long maxNumber = q - 1;
    std::uniform_int_distribution<int> dis(minNumber, maxNumber);

    vector<vector<unsigned long long>> A_poly(
        batch, vector<unsigned long long>(n * 2));
    vector<vector<unsigned long long>> B_poly(
        batch, vector<unsigned long long>(n * 2));

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i] = dis(gen);
            B_poly[j][i] = dis(gen);
        }
    }

    // Zero Pad
    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n; i++)
        {
            A_poly[j][i + n] = 0;
            B_poly[j][i + n] = 0;
        }
    }

    std::vector<std::vector<COMPLEX_C>> A_vec(batch,
                                              std::vector<COMPLEX_C>(n * 2));
    std::vector<std::vector<COMPLEX_C>> B_vec(batch,
                                              std::vector<COMPLEX_C>(n * 2));

    std::vector<std::vector<COMPLEX_C>> vec_GPU(
        2 * batch, std::vector<COMPLEX_C>(n * 2));  // A and B together

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX_C A_element = A_poly[j][i];
            A_vec[j][i] = A_element;

            COMPLEX_C B_element = B_poly[j][i];
            B_vec[j][i] = B_element;
        }
    }

    for (int j = 0; j < batch; j++)
    {  // LOAD A
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX_C element = A_poly[j][i];
            vec_GPU[j][i] = element;
        }
    }

    for (int j = 0; j < batch; j++)
    {  // LOAD B
        for (int i = 0; i < n * 2; i++)
        {
            COMPLEX_C element = B_poly[j][i];
            vec_GPU[j + batch][i] = element;
        }
    }

    FFT fft_generator(n);

    for (int j = 0; j < batch; j++)
    {
        fft_generator.fft(A_vec[j]);
        fft_generator.fft(B_vec[j]);
    }

    for (int j = 0; j < batch; j++)
    {
        for (int i = 0; i < n * 2; i++)
        {
            A_vec[j][i] = A_vec[j][i] * B_vec[j][i];
        }
    }

    for (int j = 0; j < batch; j++)
    {
        fft_generator.ifft(A_vec[j]);
    }

    /////////////////////////////////////////////////////////////////////////

    COMPLEX* Forward_InOut_Datas;

    THROW_IF_CUDA_ERROR(cudaMalloc(
        &Forward_InOut_Datas,
        2 * batch * n * 2 * sizeof(COMPLEX)));  // 2 --> A and B, batch -->
                                                // batch size, 2 --> zero pad

    for (int j = 0; j < 2 * batch; j++)
    {
        THROW_IF_CUDA_ERROR(
            cudaMemcpy(Forward_InOut_Datas + (n * 2 * j), vec_GPU[j].data(),
                       n * 2 * sizeof(COMPLEX), cudaMemcpyHostToDevice));
    }
    /////////////////////////////////////////////////////////////////////////

    COMPLEX* Root_Table_Device;

    THROW_IF_CUDA_ERROR(cudaMalloc(&Root_Table_Device, n * sizeof(COMPLEX)));

    vector<COMPLEX_C> reverse_table = fft_generator.ReverseRootTable();
    THROW_IF_CUDA_ERROR(cudaMemcpy(Root_Table_Device, reverse_table.data(),
                                   n * sizeof(COMPLEX),
                                   cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    COMPLEX* Inverse_Root_Table_Device;

    THROW_IF_CUDA_ERROR(
        cudaMalloc(&Inverse_Root_Table_Device, n * sizeof(COMPLEX)));

    vector<COMPLEX_C> inverse_reverse_table =
        fft_generator.InverseReverseRootTable();
    THROW_IF_CUDA_ERROR(
        cudaMemcpy(Inverse_Root_Table_Device, inverse_reverse_table.data(),
                   n * sizeof(COMPLEX), cudaMemcpyHostToDevice));

    /////////////////////////////////////////////////////////////////////////

    ntt_configuration cfg_fft = {.n_power = (logn + 1),
                                 .ntt_type = FORWARD,
                                 .zero_padding = false,
                                 .stream = 0};
    GPU_NTT(Forward_InOut_Datas, Root_Table_Device, cfg_fft, batch * 2, false);

    ntt_configuration cfg_ifft = {
        .n_power = (logn + 1),
        .ntt_type = INVERSE,
        .zero_padding = false,
        .mod_inverse = COMPLEX(fft_generator.n_inverse, 0.0),
        .stream = 0};
    GPU_NTT(Forward_InOut_Datas, Inverse_Root_Table_Device, cfg_ifft, batch,
            true);

    //
    COMPLEX test[batch * 2 * n];
    THROW_IF_CUDA_ERROR(cudaMemcpy(test, Forward_InOut_Datas,
                                   batch * n * 2 * sizeof(COMPLEX),
                                   cudaMemcpyDeviceToHost));

    for (int j = 0; j < batch; j++)
    {
        vector<unsigned long long> test_school =
            schoolbook_poly_multiplication(A_poly[j], B_poly[j], q, n);
        for (int i = 0; i < n * 2; i++)
        {
            signed gpu_result = std::round(test[(j * (n * 2)) + i].real());
            signed cpu_result = std::round(A_vec[j][i].real());

            if (test_school[i] != (gpu_result % q))
            {
                throw("ERROR");
            }

            if (i < 10)
            {
                cout << test_school[i] << " - " << gpu_result % q << endl;
            }
        }
    }

    return EXIT_SUCCESS;
}

/*
// tests



cmake -D CMAKE_CUDA_ARCHITECTURES=86 -D COPLEX_DATA_TYPE=0 -B./cmake-build
cmake --build ./cmake-build/ --target fft_test --parallel

./cmake-build/fft_test 12 1

*/
