"""
How would one figure out a how many tanks are in a bag if you are
allowed to randomly draw n of tham and know, that each has a number
1,...,N, where N is the total number, attached.
"""
import math

import matplotlib.pyplot as plt
import numpy as np

Ntrue = 300


def get_prior_for_N(N: int) -> np.ndarray:
    """
    N: quesses total number of tanks
    """
    return np.ones(N) / N


def draw_tank(drawn: np.ndarray | None = None) -> int:
    while True:
        obs = np.random.randint(1, Ntrue + 1)
        if drawn is None:
            return obs

        if obs not in drawn:
            return obs


def p_obs(obs: np.ndarray, Nprior: np.ndarray) -> float:
    """
    obs: observed data

    p(obs) = sum_N p(obs|N) * p(N)
    """

    Nquess = (Nprior != 0).sum()
    pobs_given_N = np.array([p_obs_given_N(obs, N) for N in range(1, len(Nprior) + 1)])
    pN = Nprior

    return (pobs_given_N * pN).sum()


def p_obs_given_N(obs: np.ndarray, Nquess: int) -> float:
    """
    obs: observed data
    N: total number of tanks

    p(obs|N) = (1 / N)^len(obs)
    """

    if any(obs > Nquess):
        return 0

    return (1 / Nquess) ** len(obs) * math.factorial(len(obs))


def update(N: int, obs: np.ndarray, Nprior: np.ndarray) -> np.ndarray:
    """
    prior: prior distribution
    obs: observed data

    p(N|obs) = p(obs|N) * p(N) / p(obs)
    """

    pobs = p_obs(obs, Nprior)
    pobsgivenN = p_obs_given_N(obs, N)
    pN = Nprior[N - 1]

    return pobsgivenN * pN / pobs


def expected_value(Nprior: np.ndarray) -> float:
    """
    Nprior: posterior distribution

    E[N] = sum_N N * p(N)
    """

    return np.arange(1, len(Nprior) + 1) @ Nprior


def main():
    K = 6
    Nquess = 500
    obs_l = []
    obs = None
    Nprior = get_prior_for_N(Nquess)
    posteriors = np.zeros((K, Nquess))

    for i in range(K):
        obs_l.append(draw_tank(obs))
        obs = np.array(obs_l)

        Nprior = np.array([update(N, obs, Nprior) for N in range(1, Nquess + 1)])
        posteriors[i] = Nprior

    obs_l = list(map(str, obs_l))

    plt.plot(get_prior_for_N(Nquess), label="Prior p(N)", color="black", linestyle="--")
    for i in range(K):
        if i % 2 == 0:
            continue

        expected = expected_value(posteriors[i])
        plt.plot(posteriors[i, :], label=f"Post p(N|obs={', '.join(obs_l[:i+1])})")
        print(f"Expected value for N={i+1}: {expected:.2f}")
        print(posteriors[i, int(expected)])

        if i == K - 1:
            y = posteriors[i, int(expected)]
            plt.annotate(
                f"E[N]={expected:.0f}",
                (expected, 0),
                (expected, y),
                rotation=40,
                horizontalalignment="center",
                verticalalignment="bottom",
                arrowprops={"arrowstyle": "->"},
            )

    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.legend(frameon=False)
    plt.xlabel("N")
    plt.ylabel("P(N|obs)")
    plt.show()


if __name__ == "__main__":
    main()
