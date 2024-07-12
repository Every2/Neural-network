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

std::vector<double> Network::feedfoward(std::vector<double> &a) {
    std::vector output{a};
    for (int l{1}; l < number_of_layers; ++l) {
        std::vector input{output};
        output.clear();
        for (int i{0}; i < _sizes.at(i); ++i) {
            for (int j{0}; j < _sizes.at(l - 1); ++j) {
                output.push_back(sigmoid(_weights[l][i][j] * input.at(j) + _biases[i][j]));
            }
        }
    }
    return output;
}

void Network::sgd(std::vector<std::pair<std::vector<double>, std::vector<double>>>& training_data, int epochs,
                  int mini_batch_size, double eta) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::size_t n{training_data.size()};
    for (auto i{0}; i < epochs; ++i) {
        std::shuffle(training_data.begin(), training_data.end(), g);
        for (auto j{0}; j < n; j += mini_batch_size) {
            update_mini_batch(training_data, j, j + mini_batch_size, eta);
        }
        std::cout << "epoch " << i << " complete" << '\n';
    }
}

void Network::update_mini_batch(std::vector<std::pair<std::vector<double>, std::vector<double>>> &training_data,
                                std::size_t begin, std::size_t end, double eta) {
    std::vector<std::vector<double>> nabla_b{};
    std::vector<std::vector<std::vector<double>>> nabla_w{};
    number_of_layers = _sizes.size();
    for (std::size_t i{1}; i < number_of_layers; ++i) {
        std::vector aux(_sizes.at(i), 0.0);
        nabla_b.push_back(aux);
    }

    for (int i{1}; i < number_of_layers; ++i) {
        std::vector<std::vector<double>> first_aux{};
        for (int j{0}; j < _sizes.at(i); ++j) {
            std::vector second_aux(_sizes.at(i - 1), 0.0);
            first_aux.push_back(second_aux);
        }
        nabla_w.push_back(first_aux);
    }

    for (auto i{begin}; i < end; ++i) {
        auto delta{backprop(training_data.at(i))};
        for (auto j{1}; j < number_of_layers; ++j) {
            for (auto k{0}; k < _sizes.at(j); ++k) {
                nabla_b[j][k] += delta.first[j][k];
            }
        }

        for (auto j{1}; j < number_of_layers; ++j) {
            for (auto k{0}; k < _sizes.at(j); ++k) {
                for (auto l{0}; l < _sizes.at(j - 1); ++l) {
                    nabla_w[j][k][l] += delta.second[j][k][l];
                }
            }
        }
    }
    double length{static_cast<double>(end - begin)};
    for (std::size_t i{1}; i < number_of_layers; ++i) {
        for (std::size_t j{0}; j < _sizes.at(i); ++j) {
            _biases[i][j] -= (eta / length) * nabla_b[i][j];
        }
    }

    for (int i{1}; i < number_of_layers; ++i) {
        for (int j{0}; j < _sizes.at(i); ++j) {
            for (int k{0}; k < _sizes.at(i - 1); ++k) {
                _weights[i][j][k] -= (eta / length) * nabla_w[i][j][k];
            }
        }
    }
}
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>>
Network::backprop(std::pair<std::vector<double>, std::vector<double>> xy) {

}

