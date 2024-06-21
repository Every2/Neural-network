#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>

class Network {

public:
    Network(std::vector<int>& sizes);
    double sigmod(double z);
private:
    std::vector<int> _sizes{};
    std::size_t numbers_of_layers;
    std::vector<std::vector<double>> _biases{};
    std::vector<std::vector<std::vector<double>>> _weights{};
};
#endif //NETWORK_HPP
