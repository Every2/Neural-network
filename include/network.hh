#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
#include <algorithm>
#include <iostream>
double sigmoid(double z);

class Network {
public:
    Network(std::vector<int>& sizes);
    std::vector<double> feedfoward(std::vector<double>& a);
    void sgd(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_data,
             int epochs, int mini_batch_size, double eta);
    void update_mini_batch(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_data,
                            std::size_t begin, std::size_t end, double eta);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>> backprop(
        std::pair<std::vector<double>, std::vector<double>> xy
    );

private:
    std::vector<int> _sizes{};
    std::size_t number_of_layers;
    std::vector<std::vector<double>> _biases{};
    std::vector<std::vector<std::vector<double>>> _weights{};
};
#endif //NETWORK_HPP
