# fun_info.py
import numpy as np

def Ufun(x, a, k, m):
    return k * ((x-a)**m) * (x > a) + k * ((-x-a)**m) * (x < (-a))

def F1(x):
    return np.sum(x**2)

def F2(x):
    return np.sum(np.abs(x)) + np.prod(np.abs(x))

def F3(x):
    dimension = len(x)
    return sum([np.sum(x[0:i+1])**2 for i in range(dimension)])

def F4(x):
    return np.max(np.abs(x))

def F5(x):
    dimension = len(x)
    return np.sum(100 * (x[1:dimension] - x[0:dimension-1]**2)**2 + (x[0:dimension-1] - 1)**2)

def F6(x):
    return np.sum(np.floor(x + 0.5)**2)

def F7(x):
    dimension = len(x)
    return np.sum([(i+1) * x[i]**4 for i in range(dimension)]) + np.random.random()

def F8(x):
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def F9(x):
    dimension = len(x)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) + 10 * dimension

def F10(x):
    dimension = len(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / dimension)) - \
           np.exp(np.sum(np.cos(2 * np.pi * x)) / dimension) + 20 + np.exp(1)

def F11(x):
    dimension = len(x)
    return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, dimension+1)))) + 1

def F12(x):
    dimension = len(x)
    return (np.pi / dimension) * (10 * np.sin(np.pi * (1 + (x[0] + 1) / 4))**2 +
           np.sum(((x[0:dimension-1] + 1) / 4)**2 * (1 + 10 * np.sin(np.pi * (1 + (x[1:dimension] + 1) / 4))**2)) +
           ((x[dimension-1] + 1) / 4)**2) + np.sum(Ufun(x, 10, 100, 4))

def F13(x):
    dimension = len(x)
    return 0.1 * (np.sin(3 * np.pi * x[0])**2 +
           np.sum((x[0:dimension-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:dimension])**2)) +
           ((x[dimension-1] - 1)**2) * (1 + np.sin(2 * np.pi * x[dimension-1])**2)) + \
           np.sum(Ufun(x, 5, 100, 4))

def F14(x):
    aS = np.array([[-32, -16, 0, 16, 32] * 5,
                   [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5])
    bS = np.zeros(25)
    for j in range(25):
        bS[j] = np.sum((x - aS[:, j])**6)
    return (1/500 + np.sum(1/(np.arange(1, 26) + bS)))**(-1)

def F15(x):
    aK = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627, 0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    bK = 1/np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    return np.sum((aK - ((x[0] * (bK**2 + x[1] * bK)) / (bK**2 + x[2] * bK + x[3])))**2)

def F16(x):
    return 4 * (x[0]**2) - 2.1 * (x[0]**4) + (x[0]**6) / 3 + x[0] * x[1] - 4 * (x[1]**2) + 4 * (x[1]**4)

def F17(x):
    return (x[1] - (x[0]**2) * 5.1/(4 * (np.pi**2)) + 5/np.pi * x[0] - 6)**2 + \
           10 * (1 - 1/(8 * np.pi)) * np.cos(x[0]) + 10

def F18(x):
    return (1 + (x[0] + x[1] + 1)**2 * (19 - 14 * x[0] + 3 * (x[0]**2) - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1]**2)) * \
           (30 + (2 * x[0] - 3 * x[1])**2 * (18 - 32 * x[0] + 12 * (x[0]**2) + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1]**2)))

def F19(x):
    aH = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.3689, 0.117, 0.2673],
                   [0.4699, 0.4387, 0.747],
                   [0.1091, 0.8732, 0.5547],
                   [0.03815, 0.5743, 0.8828]])
    return -np.sum([cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2))) for i in range(4)])

def F20(x):
    aH = np.array([[10, 3, 17, 3.5, 1.7, 8],
                   [0.05, 10, 17, 0.1, 8, 14],
                   [3, 3.5, 1.7, 10, 17, 8],
                   [17, 8, 0.05, 10, 0.1, 14]])
    cH = np.array([1, 1.2, 3, 3.2])
    pH = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                   [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                   [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
                   [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    return -np.sum([cH[i] * np.exp(-np.sum(aH[i, :] * ((x - pH[i, :])**2))) for i in range(4)])

def F21(x):
    aSH = np.array([[4, 4, 4, 4],
                    [1, 1, 1, 1],
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 5, 3, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    return -np.sum([1/((np.sum((x - aSH[i, :])**2) + cSH[i])) for i in range(5)])

def F22(x):
    aSH = np.array([[4, 4, 4, 4],
                    [1, 1, 1, 1],
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 5, 3, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    return -np.sum([1/((np.sum((x - aSH[i, :])**2) + cSH[i])) for i in range(7)])

def F23(x):
    aSH = np.array([[4, 4, 4, 4],
                    [1, 1, 1, 1],
                    [8, 8, 8, 8],
                    [6, 6, 6, 6],
                    [3, 7, 3, 7],
                    [2, 9, 2, 9],
                    [5, 5, 3, 3],
                    [8, 1, 8, 1],
                    [6, 2, 6, 2],
                    [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    return -np.sum([1/((np.sum((x - aSH[i, :])**2) + cSH[i])) for i in range(10)])

def F24(x, y_true, y_pred):
    """XGBoost超参数优化的自定义目标函数"""
    import numpy as np

    try:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        N = len(y_true)
        term1 = (1 / (3 * N)) * np.sum((y_true - y_pred) ** 2)  # MSE
        term2 = (1 / (3 * N)) * np.sum(np.abs(y_true - y_pred))  # MAE
        term3 = (1 / (3 * N)) * np.sum(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-10))  # MAPE

        result = float(term1 + term2 + term3)
        print(f"Calculated result: {result}")

        return result

    except Exception as e:
        print(f"Error in F24: {e}")
        import traceback
        traceback.print_exc()
        raise

def get_function_info(F):
    """Returns the function information based on the function name"""
    function_dict = {
        'F1': (F1, -100, 100, 30),
        'F2': (F2, -10, 10, 30),
        'F3': (F3, -100, 100, 30),
        'F4': (F4, -100, 100, 30),
        'F5': (F5, -30, 30, 30),
        'F6': (F6, -100, 100, 30),
        'F7': (F7, -1.28, 1.28, 30),
        'F8': (F8, -500, 500, 30),
        'F9': (F9, -5.12, 5.12, 30),
        'F10': (F10, -32, 32, 30),
        'F11': (F11, -600, 600, 30),
        'F12': (F12, -50, 50, 30),
        'F13': (F13, -50, 50, 30),
        'F14': (F14, -65.536, 65.536, 2),
        'F15': (F15, -5, 5, 4),
        'F16': (F16, -5, 5, 2),
        'F17': (F17, [-5, 0], [10, 15], 2),
        'F18': (F18, -2, 2, 2),
        'F19': (F19, 0, 1, 3),
        'F20': (F20, 0, 1, 6),
        'F21': (F21, 0, 10, 4),
        'F22': (F22, 0, 10, 4),
        'F23': (F23, 0, 10, 4),
        'F24': (F24, [0.01, 10], [1.0, 1000], 2)
    }

    if F in function_dict:
        fitness, lowerbound, upperbound, dimension = function_dict[F]
        return fitness, lowerbound, upperbound, dimension
    else:
        raise ValueError(f"Function {F} not found")