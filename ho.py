# ho.py
import numpy as np
import cupy as cp
from levy import levy_gpu


def check_bounds_gpu(positions, lower_bounds, upper_bounds):
    try:
        with cp.cuda.Device(0):
            positions_gpu = cp.asarray(positions)
            lower_bounds_gpu = cp.asarray(lower_bounds)
            upper_bounds_gpu = cp.asarray(upper_bounds)
            positions_gpu = cp.minimum(cp.maximum(positions_gpu, lower_bounds_gpu), upper_bounds_gpu)
            return cp.asnumpy(positions_gpu)
    except:
        return np.minimum(np.maximum(positions, lower_bounds), upper_bounds)


def check_and_convert_to_array(x):
    try:
        if not isinstance(x, np.ndarray):
            return np.array(x, dtype=np.float64)
        return x
    except Exception as e:
        print(f"Error converting to numpy array: {e}")
        raise


def HO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness, *args):
    try:
        if isinstance(lowerbound, (int, float)):
            lowerbound = np.ones(dimension) * lowerbound
        if isinstance(upperbound, (int, float)):
            upperbound = np.ones(dimension) * upperbound

        lowerbound = check_and_convert_to_array(lowerbound)
        upperbound = check_and_convert_to_array(upperbound)

        X = np.zeros((SearchAgents, dimension))
        for i in range(dimension):
            X[:, i] = lowerbound[i] + np.random.rand(SearchAgents) * (upperbound[i] - lowerbound[i])

        fit = np.zeros(SearchAgents)
        for i in range(SearchAgents):
            if args:
                fit[i] = fitness(X[i].copy(), *args)
            else:
                fit[i] = fitness(X[i].copy())

        best_so_far = np.zeros(Max_iterations)
        best_idx = np.argmin(fit)
        fbest = fit[best_idx]
        Xbest = X[best_idx].copy()

        for t in range(Max_iterations):
            for i in range(SearchAgents // 2):
                Dominant_hippopotamus = check_and_convert_to_array(Xbest.copy())
                I1 = np.random.randint(1, 3)
                I2 = np.random.randint(1, 3)
                Ip1 = np.random.randint(0, 2, 2)
                RandGroupNumber = np.random.randint(1, SearchAgents + 1)
                RandGroup = np.random.permutation(SearchAgents)[:RandGroupNumber]

                MeanGroup = np.mean(X[RandGroup], axis=0) if len(RandGroup) > 1 else X[RandGroup[0]]
                MeanGroup = check_and_convert_to_array(MeanGroup)

                Alfa = {
                    1: I2 * np.random.rand(dimension) + (1 - Ip1[0]),
                    2: 2 * np.random.rand(dimension) - 1,
                    3: np.random.rand(dimension),
                    4: I1 * np.random.rand(dimension) + (1 - Ip1[1]),
                    5: np.random.rand()
                }

                A = Alfa[np.random.randint(1, 6)]
                B = Alfa[np.random.randint(1, 6)]
                A = check_and_convert_to_array(A)
                B = check_and_convert_to_array(B)

                X_P1 = X[i] + np.random.rand() * (Dominant_hippopotamus - I1 * X[i])
                T = np.exp(-t / Max_iterations)

                if T > 0.6:
                    X_P2 = X[i] + A * (Dominant_hippopotamus - I2 * MeanGroup)
                else:
                    if np.random.rand() > 0.5:
                        X_P2 = X[i] + B * (MeanGroup - Dominant_hippopotamus)
                    else:
                        X_P2 = np.random.rand(dimension) * (upperbound - lowerbound) + lowerbound

                X_P1 = check_bounds_gpu(X_P1, lowerbound, upperbound)
                X_P2 = check_bounds_gpu(X_P2, lowerbound, upperbound)

                if args:
                    F_P1 = fitness(X_P1, *args)
                    F_P2 = fitness(X_P2, *args)
                else:
                    F_P1 = fitness(X_P1)
                    F_P2 = fitness(X_P2)

                if F_P1 < fit[i]:
                    X[i] = X_P1.copy()
                    fit[i] = F_P1

                if F_P2 < fit[i]:
                    X[i] = X_P2.copy()
                    fit[i] = F_P2

            for i in range(SearchAgents // 2, SearchAgents):
                predator = lowerbound + np.random.rand(dimension) * (upperbound - lowerbound)
                predator = check_and_convert_to_array(predator)

                if args:
                    F_HL = fitness(predator, *args)
                else:
                    F_HL = fitness(predator)

                distance2Leader = np.abs(predator - X[i])
                distance2Leader = check_and_convert_to_array(distance2Leader)

                b = np.random.uniform(2, 4)
                c = np.random.uniform(1, 1.5)
                d = np.random.uniform(2, 3)
                l = np.random.uniform(-2 * np.pi, 2 * np.pi)

                RL = 0.05 * levy_gpu(SearchAgents, dimension, 1.5)
                RL = check_and_convert_to_array(RL)

                if fit[i] > F_HL:
                    X_P3 = RL[i] * predator + (b / (c - d * np.cos(l))) * (1 / distance2Leader)
                else:
                    rand_values = check_and_convert_to_array(np.random.rand(dimension))
                    X_P3 = RL[i] * predator + (b / (c - d * np.cos(l))) * (1 / (2 * distance2Leader + rand_values))

                X_P3 = check_bounds_gpu(X_P3, lowerbound, upperbound)

                if args:
                    F_P3 = fitness(X_P3, *args)
                else:
                    F_P3 = fitness(X_P3)

                if F_P3 < fit[i]:
                    X[i] = X_P3.copy()
                    fit[i] = F_P3

            for i in range(SearchAgents):
                LO_LOCAL = lowerbound / (t + 1)
                HI_LOCAL = upperbound / (t + 1)
                LO_LOCAL = check_and_convert_to_array(LO_LOCAL)
                HI_LOCAL = check_and_convert_to_array(HI_LOCAL)

                Alfa = {
                    1: 2 * np.random.rand(dimension) - 1,
                    2: np.random.rand(),
                    3: np.random.randn()
                }
                D = Alfa[np.random.randint(1, 4)]
                D = check_and_convert_to_array(D) if isinstance(D, np.ndarray) else D

                X_P4 = X[i] + np.random.rand() * (LO_LOCAL + D * (HI_LOCAL - LO_LOCAL))

                X_P4 = check_bounds_gpu(X_P4, lowerbound, upperbound)

                if args:
                    F_P4 = fitness(X_P4, *args)
                else:
                    F_P4 = fitness(X_P4)

                if F_P4 < fit[i]:
                    X[i] = X_P4.copy()
                    fit[i] = F_P4

                    if fit[i] < fbest:
                        fbest = fit[i]
                        Xbest = X[i].copy()

            best_so_far[t] = fbest
            print(f'Iteration {t + 1}: Best value = {fbest}')

        return fbest, Xbest, best_so_far

    except Exception as e:
        print(f"HO algorithm error: {e}")
        import traceback
        traceback.print_exc()
        raise