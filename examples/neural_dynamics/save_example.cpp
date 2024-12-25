#include <iostream>
#include <torch/torch.h>

int main() {
    // 1. Create some data in a dict
    torch::nn::Linear model(3, 4);
    torch::save(model, "model.pt");
    try {
        torch::save(dataset_map, filename);
        std::cout << "Dataset saved to " << filename << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error saving the dataset: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
