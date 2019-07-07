#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <random>

#include "../include/matplotlibcpp.h"

const size_t v_size = 10;

namespace plt = matplotlibcpp;

// display the vectors
void print_list(const std::string &message, std::vector<double> &v) {
    std::cout << std::left << std::setw(25) << message;
    for (auto &item:v)
        std::cout << std::fixed << std::setprecision(2) << std::right << std::setw(10) << item << " ";
    std::cout << std::endl;
}

float sensor_noise(float variance) {
    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_real_distribution<> distr(0.0, 1.0); // define the range

    return 2 * variance * distr(eng) - variance;
}

int main() {

    float a{0.75};
    float b{0.5};
    float c{1.0};
    float r{100.0};   //sensor's noise
    size_t i;


    std::vector<double> x_k(v_size);    // The real values
    std::vector<double> z_k(v_size);    // The observations
    std::vector<double> p_k(v_size);    // The prediction error
    std::vector<double> g_k(v_size);    // The Kalman gain
    std::vector<double> x_hat(v_size);  // The state estimation
    std::vector<double> u_k(v_size);    // The control signal
    std::vector<double> v_k(v_size);    // The noise signal

    // initialisation
    for (i = 0, x_k[0] = 1000, z_k[0] = 1000 + sensor_noise(r); i < v_size; ++i) {
        u_k[i] = i;
        v_k[i] = sensor_noise(r);

        // update state
        x_k[i + 1] = a * x_k[i] + b * u_k[i];

        // update observation
        z_k[i + 1] = c * x_k[i + 1] + v_k[i];
    }

    print_list("The real values:", x_k);
    print_list("The observed values:", z_k);


    // Kalman filter
    for (p_k[0] = 1.0, x_hat[0] = z_k[0], i = 0; i < x_k.size(); i++) {
        //predict
        x_hat[i + 1] = a * x_hat[i] + b * u_k[i + 1];
        p_k[i + 1] = a * p_k[i] * a;

        //update
        g_k[i] = (p_k[i + 1] * c) / (c * p_k[i + 1] * c + r);
        x_hat[i + 1] = x_hat[i + 1] + g_k[i] * (z_k[i + 1] - c * x_hat[i + 1]);
        p_k[i + 1] = (1 - g_k[i] * c) * p_k[i + 1];
    }

    print_list("The estimated values:", x_hat);

    plt::plot(x_k, "g");
    plt::named_plot("real values", x_k);
    plt::plot(z_k, "b");
    plt::named_plot("observations", z_k);
    plt::plot(x_hat, "r");
    plt::named_plot("state estimation", x_hat);
    plt::legend();
    plt::show();

    return 0;
}