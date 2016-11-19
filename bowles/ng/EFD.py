__author__ = 'joe'
import numpy as np
import matplotlib.pyplot as plt


# EFD Factory
def ExponentialFamilyDistribution(a, b, T):
    return lambda y, eta: b(y) * np.e ** (eta * T(y, eta) - a(eta))


# Bernoulli
a = lambda eta: 1 + np.e ** eta
b = lambda y: 1
T = lambda y, eta: y
# Expectation for Bernoulli, or h(x) or E[y|x;theta]=P(y=1|x;theta)=1/(1+e^(-theta_transposed*x))

Bernoulli_Dist = ExponentialFamilyDistribution(a, b, T)

# Gaussian
a = lambda eta: -.5 * eta ** 2
b = lambda y: (1 / np.sqrt(2 * np.pi)) * np.e ** (-.5 * y ** 2)
T = lambda y, eta: y

Gaussian_Dist = ExponentialFamilyDistribution(a, b, T)

# Multinomial
a = lambda etas: -np.log((np.e ** etas[-1]) / (1 + sum(np.e ** etas[j] for j in range(0, len(etas) - 1))))
b = lambda y: 1
T = lambda y, etas: np.array([1 if y != i else 0 for i in range(len(etas))])

Multinomial_Dist = ExponentialFamilyDistribution(a, b, T)

if __name__ == "__main__":
    # Plot on interval
    dist_min, dist_max, dist_interval = -3, 3, .01
    dist_range = [dist_min + dist_step * dist_interval for dist_step in
                  range((dist_max - dist_min) * int(1 / dist_interval) + 1)]

    bernoulli = map(lambda y: Bernoulli_Dist(y, 1), dist_range)
    gaussian = map(lambda y: Gaussian_Dist(y, 0), dist_range)
    multinomial = map(lambda y:Multinomial_Dist(y,lambda eta:),dist_range)
    plt.plot(dist_range, bernoulli, dist_range, gaussian)
    plt.show()
