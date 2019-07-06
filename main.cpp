#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <chrono>

#include <random>

#include "include/matplotlibcpp.h"

namespace plt = matplotlibcpp;

// display the vectors
void print_list(const std::string &message, std::vector<double> &v)
{
    std::cout << std::left << std::setw(25) << message;
    for (auto &item:v)
        std::cout << std::fixed << std::setprecision(2) << std::right << std::setw(10) << item << " ";
    std::cout << std::endl;
}


int main() {

    float a{0.75};
    size_t i;
    float r = 200; //sensor's noise

    std::vector<double> x_k(10);    // The real values
    std::vector<double> z_k(10);    // The observations
    std::vector<double> p_k(10);    // The prediction error
    std::vector<double> g_k(10);    // The Kalman gain
    std::vector<double> x_hat(10);  // The state estimation

    std::random_device rd;  // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(-200, 200); // define the range


    for (i = 0, x_k[0] = 1000; i < x_k.size(); i++)
    {
        x_k[i+1] = a * x_k[i]; // + distribution(generator);
    }

    print_list("The real values:", x_k);

    for (i = 0; i < x_k.size(); i++)
    {
        z_k[i] = x_k[i] + distr(eng);
    }

    print_list("The observed values:", z_k);

    for (p_k[0] = 1.0, x_hat[0] = z_k[0], i = 0; i < x_k.size(); i++)
    {
        //predict
        x_hat[i+1] = a * x_hat[i];
        p_k[i+1] = a * p_k[i] * a;

        //update
        g_k[i] = p_k[i+1]/(p_k[i+1] + r);
        x_hat[i+1] = x_hat[i+1] + g_k[i] * (z_k[i+1] - x_hat[i+1]);
        p_k[i+1] = (1 - g_k[i]) * p_k[i+1];
    }

    print_list("The estimated values:", x_hat);

    plt::plot(x_k);
    plt::plot(z_k);
    plt::plot(x_hat);
    plt::show();

    return 0;
}