#include <iostream>
#include <chrono>
#include <torch/torch.h>
#include <Eigen/Dense>

int main() {
    // Define large, mid and small matrix dimensions
    int large_rows = 1000;
    int large_cols = 1000;
    int mid_rows = 100;
    int mid_cols = 100;
    int small_rows = 10;
    int small_cols = 10;

    // Create large, mid and small torch::Tensors and Eigen matrices
    torch::Tensor large_torch_tensor = torch::rand({large_rows, large_cols});
    torch::Tensor mid_torch_tensor = torch::rand({mid_rows, mid_cols});
    torch::Tensor small_torch_tensor = torch::rand({small_rows, small_cols});
    Eigen::MatrixXd large_eigen_matrix = Eigen::MatrixXd::Random(large_rows, large_cols);
    Eigen::MatrixXd mid_eigen_matrix = Eigen::MatrixXd::Random(mid_rows, mid_cols);
    Eigen::MatrixXd small_eigen_matrix = Eigen::MatrixXd::Random(small_rows, small_cols);

    // Get the default device (CPU or CUDA if available)
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    // Print the default device
    std::cout << "Default device: " << device << std::endl;

    // Move torch::Tensors to the default device
    large_torch_tensor = large_torch_tensor.to(device);
    mid_torch_tensor = mid_torch_tensor.to(device);
    small_torch_tensor = small_torch_tensor.to(device);

    // Benchmark large torch::Tensor matrix multiplication
    auto start_large_torch_matmul = std::chrono::high_resolution_clock::now();
    // pre-allocate memory for the result
    torch::Tensor large_torch_result = torch::empty({large_rows, large_rows}, device);
    large_torch_result = torch::matmul(large_torch_tensor, large_torch_tensor.transpose(0, 1));
    auto end_large_torch_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_large_torch_matmul = end_large_torch_matmul - start_large_torch_matmul;

    // Benchmark large Eigen matrix multiplication
    auto start_large_eigen_matmul = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd large_eigen_result = large_eigen_matrix * large_eigen_matrix.transpose();
    auto end_large_eigen_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_large_eigen_matmul = end_large_eigen_matmul - start_large_eigen_matmul;

    // Benchmark mid torch::Tensor matrix multiplication
    auto start_mid_torch_matmul = std::chrono::high_resolution_clock::now();
    // pre-allocate memory for the result
    torch::Tensor mid_torch_result = torch::empty({mid_rows, mid_rows}, device);
    mid_torch_result = torch::matmul(mid_torch_tensor, mid_torch_tensor.transpose(0, 1));
    auto end_mid_torch_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_mid_torch_matmul = end_mid_torch_matmul - start_mid_torch_matmul;

    // Benchmark mid Eigen matrix multiplication
    auto start_mid_eigen_matmul = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd mid_eigen_result = mid_eigen_matrix * mid_eigen_matrix.transpose();
    auto end_mid_eigen_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_mid_eigen_matmul = end_mid_eigen_matmul - start_mid_eigen_matmul;

    // Benchmark small torch::Tensor matrix multiplication
    auto start_small_torch_matmul = std::chrono::high_resolution_clock::now();
    // pre-allocate memory for the result
    torch::Tensor small_torch_result = torch::empty({small_rows, small_rows}, device);
    small_torch_result = torch::matmul(small_torch_tensor, small_torch_tensor.transpose(0, 1));
    auto end_small_torch_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_small_torch_matmul = end_small_torch_matmul - start_small_torch_matmul;

    // Benchmark small Eigen matrix multiplication
    auto start_small_eigen_matmul = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd small_eigen_result = small_eigen_matrix * small_eigen_matrix.transpose();
    auto end_small_eigen_matmul = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_small_eigen_matmul = end_small_eigen_matmul - start_small_eigen_matmul;

    // Benchmark small torch::Tensor inverse
    auto start_small_torch_inverse = std::chrono::high_resolution_clock::now();
    // pre-allocate memory for the result
    torch::Tensor small_torch_inverse = torch::empty({small_rows, small_cols}, device);
    small_torch_inverse = torch::inverse(small_torch_tensor);
    auto end_small_torch_inverse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_small_torch_inverse = end_small_torch_inverse - start_small_torch_inverse;

    // Benchmark small Eigen matrix inverse
    auto start_small_eigen_inverse = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd small_eigen_inverse = small_eigen_matrix.inverse();
    auto end_small_eigen_inverse = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_small_eigen_inverse = end_small_eigen_inverse - start_small_eigen_inverse;

    // Print results
    std::cout << "Large Torch::Tensor matrix multiplication time: " << elapsed_large_torch_matmul.count() << " seconds" << std::endl;
    std::cout << "Large Eigen matrix multiplication time: " << elapsed_large_eigen_matmul.count() << " seconds" << std::endl;
    std::cout << "Mid Torch::Tensor matrix multiplication time: " << elapsed_mid_torch_matmul.count() << " seconds" << std::endl;
    std::cout << "Mid Eigen matrix multiplication time: " << elapsed_mid_eigen_matmul.count() << " seconds" << std::endl;
    std::cout << "Small Torch::Tensor matrix multiplication time: " << elapsed_small_torch_matmul.count() << " seconds" << std::endl;
    std::cout << "Small Eigen matrix multiplication time: " << elapsed_small_eigen_matmul.count() << " seconds" << std::endl;
    std::cout << "Small Torch::Tensor inverse time: " << elapsed_small_torch_inverse.count() << " seconds" << std::endl;
    std::cout << "Small Eigen matrix inverse time: " << elapsed_small_eigen_inverse.count() << " seconds" << std::endl;

    return 0;
}

// PC specs:
// - CPU: 13th Gen Intel® Core™ i7-13620H × 16
// - GPU: NVIDIA GeForce RTX 4060
// ```
// $ ./test_torch_eigen 
// Default device: cuda
// Large Torch::Tensor matrix multiplication time: 0.0187716 seconds
// Large Eigen matrix multiplication time: 0.121851 seconds
// Mid Torch::Tensor matrix multiplication time: 0.00302689 seconds
// Mid Eigen matrix multiplication time: 0.000178552 seconds
// Small Torch::Tensor matrix multiplication time: 0.00202544 seconds
// Small Eigen matrix multiplication time: 2.229e-06 seconds
// Small Torch::Tensor inverse time: 0.0824715 seconds
// Small Eigen matrix inverse time: 7.489e-06 seconds
// ```