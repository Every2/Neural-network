#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <vector>
double sigmoid(double z);

class Network {

public:
    Network(std::vector<int>& sizes);
    std::vector<double> feedfoward(std::vector<double>& a);

private:
    std::vector<int> _sizes{};
    std::size_t number_of_layers;
    std::vector<std::vector<double>> _biases{};
    std::vector<std::vector<std::vector<double>>> _weights{};
};
#endif //NETWORK_HPP
