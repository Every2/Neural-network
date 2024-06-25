#include "../include/network.hh"
#include <random>
#include <cmath>

Network::Network(std::vector<int> &sizes) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution generated_normal_distribution{0.0, 1.0};
    _sizes = sizes;
    number_of_layers = _sizes.size();
    for (std::size_t i{1}; i < number_of_layers; ++i) {
        std::vector<double> aux{};
        for (std::size_t j{0}; j < _sizes.at(i); ++j) {
            double random_double{generated_normal_distribution(gen)};
            aux.push_back(random_double);
        }
        _biases.push_back(aux);
    }

    for (int i{1}; i < number_of_layers; ++i) {
        std::vector<std::vector<double>> first_aux{};
        for (int j{0}; j < _sizes.at(i); ++j) {
            std::vector<double> second_aux{};
            for (int k{0}; k < _sizes.at(i - 1); ++k) {
                double random_double{generated_normal_distribution(gen)};
                second_aux.push_back(random_double);
            }
            first_aux.push_back(second_aux);
        }
        _weights.push_back(first_aux);
    }
}

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

std::vector<double> Network::feedfoward(std::vector<double>& a) {
    std::vector output {a};
    for (int l{1}; l < number_of_layers; ++l) {
        std::vector input {output};
        output.clear();
        for  (int i {0}; i < _sizes.at(i); ++i) {
            for (int j{0}; j < _sizes.at(l - 1); ++j) {
                output.push_back(sigmoid(_weights[l][i][j] * input.at(j) + _biases[i][j]));
            }
        }
    }
    return output;
}

