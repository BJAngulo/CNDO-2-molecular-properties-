#pragma once

#include <cmath>

// 1. Gaussian helpers 

// Gaussian product center
inline double gaussian_product_center(
    double center_A,
    double center_B,
    double alpha, 
    double beta
)
{
    return (alpha * center_A + beta * center_B) / (alpha + beta);
}

// Gaussian exponential prefactor
inline double gaussian_prefactor(
    double center_A, 
    double center_B,
    double alpha,
    double beta
)
{
    double difference = center_A - center_B;

    return std::exp(
        -(alpha * beta * difference * difference)
        / (alpha + beta)
    ); 
}

// 2. Mathematical helpers

// Factorial 
inline int factorial(int n)
{
    int result = 1;

    for (int i = 2; i <= n; i++){
        result *= i;
    }

    return result;
}

// Binomial coefficient
inline int binomial(int n, int k)
{
    return factorial(n) /
        (factorial(k) * factorial(n-k));
}

// double factorial 
inline int double_factorial(int n)
{
    int result = 1;

    for (int i = n; i > 1; i -= 2) {
        result *= i; 
    }

    return result;
}


// General analytical 1d overlap 
inline double overlap_1d(
    double center_A,
    double center_B,
    double alpha,
    double beta,
    int l_A,
    int l_B
)
{
    double p = alpha + beta;
    
    double XP = gaussian_product_center(
        center_A, center_B, alpha, beta
    );

    double K = gaussian_prefactor(center_A, center_B, alpha, beta);

    double sum = 0.0;

    for (int i= 0; i <= l_A; i++) {
        for (int j = 0; j <= l_B; j++) {

            // Odd powers integrate to zero 
            if ((i + j) % 2 != 0) {
                continue;
            }

            double coefficient = 
                binomial(l_A, i) * binomial(l_B, j);

            double distance_A = std::pow(XP - center_A, l_A - i);
            double distance_B = std::pow(XP - center_B, l_B -j);
            
            double moment = 
                double_factorial(i + j - 1)/
                std::pow(2.0 * p, (i + j) / 2.0);
            
            double term = coefficient * distance_A * distance_B * moment;

            sum += term; 

        }
    }

    return K * std::sqrt(std::acos(-1.0) / p) * sum;
}

// Analytical derivative of 1D overlap with respect to Gaussian center A
inline double overlap_1d_derivative(
    double center_A,
    double center_B,
    double alpha,
    double beta,
    int l_A,
    int l_B
)
{
    double lower_term = 0.0;

    // First term: -l_A * S(l_A - 1, l_B)
    if (l_A > 0) {
        lower_term = -l_A * overlap_1d(
            center_A, 
            center_B,
            alpha, 
            beta,
            l_A - 1,
            l_B
        );
    }

    // Second term: 2 * alpha * S(l_A + 1, l_B)
    double upper_term = 2.0 * alpha * overlap_1d(
        center_A,
        center_B, 
        alpha, 
        beta, 
        l_A + 1, 
        l_B
    );

    return lower_term + upper_term;
}
