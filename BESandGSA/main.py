import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
import time

# ------------------------------------------------------------
#  Test fonksiyonları 
# ------------------------------------------------------------
class TestFunctions:
    @staticmethod
    def sphere(x: np.ndarray):
        return np.sum(x ** 2)

    @staticmethod
    def rosenbrock(x: np.ndarray):
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2)

    @staticmethod
    def ackley(x: np.ndarray):
        a, b, c = 20, 0.2, 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x ** 2) / len(x)))
        term2 = -np.exp(np.sum(np.cos(c * x)) / len(x))
        return term1 + term2 + a + np.exp(1)

    @staticmethod
    def rastrigin(x: np.ndarray):
        n = len(x)
        return 10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))

    @staticmethod
    def griewank(x: np.ndarray):
        sum_sq = np.sum(x ** 2 / 4000)
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq - prod_cos


# ------------------------------------------------------------
#  Golden Sine Algorithm 
# ------------------------------------------------------------
class GoldenSineAlgorithm:
    def __init__(self, obj_func, n_dim, lb, ub,
                 population_size=30, max_iter=100):
        self.f = obj_func
        self.D = n_dim
        self.lb = np.array([lb] * n_dim) if np.isscalar(lb) else lb.astype(float)
        self.ub = np.array([ub] * n_dim) if np.isscalar(ub) else ub.astype(float)
        self.N = population_size
        self.T = max_iter
        self.tau = (np.sqrt(5) - 1) / 2.0  # golden ratio

    # --------------------------------------------------
    def _clip(self, P):
        return np.clip(P, self.lb, self.ub)

    # --------------------------------------------------
    def optimize(self):
        # init pop
        P = np.random.uniform(self.lb, self.ub, (self.N, self.D))
        fit = np.array([self.f(x) for x in P])
        gbest = P[np.argmin(fit)].copy()
        best_y = fit.min()
        history = [best_y]

        # golden‑section scalar interval (for sine step)
        a_int, b_int = -np.pi, np.pi
        x1 = a_int * self.tau + b_int * (1 - self.tau)
        x2 = a_int * (1 - self.tau) + b_int * self.tau

        for _ in range(self.T):
            r1 = 2 * np.pi * np.random.rand(self.N, 1)
            r2 = np.pi * np.random.rand(self.N, 1)
            abs_sin = np.abs(np.sin(r1))
            sin_r1 = np.sin(r1)

            for i in range(self.N):
                P[i] = (P[i] * abs_sin[i] -
                        r2[i] * sin_r1[i] *
                        np.abs(x1 * gbest - x2 * P[i]))
            P = self._clip(P)

            # fitness
            fit = np.array([self.f(x) for x in P])
            if fit.min() < best_y:
                gbest = P[np.argmin(fit)].copy()
                best_y = fit.min()

            # golden‑section scalar update (simple)
            if np.random.rand() < 0.5:
                b_int = x2; x2 = x1
                x1 = a_int * self.tau + b_int * (1 - self.tau)
            else:
                a_int = x1; x1 = x2
                x2 = a_int * (1 - self.tau) + b_int * self.tau

            history.append(best_y)
        return gbest, best_y, history


# ------------------------------------------------------------
# Bald Eagle Search Algoirthm
# ------------------------------------------------------------
class BaldEagle:
    def __init__(self, obj_func, n_dim, lb, ub,
                 pop_size=30, max_iter=100,
                 alpha=2.0, a_spiral=8.0, R_spiral=1.0,
                 c1=1.5, c2=1.5):
        self.f = obj_func
        self.D = n_dim
        self.lb = np.array([lb] * n_dim) if np.isscalar(lb) else lb.astype(float)
        self.ub = np.array([ub] * n_dim) if np.isscalar(ub) else ub.astype(float)
        self.N, self.T = pop_size, max_iter
        self.alpha, self.a_spiral, self.R_spiral = alpha, a_spiral, R_spiral
        self.c1, self.c2 = c1, c2

    # -------------------- helpers
    def _clip(self, P):
        return np.clip(P, self.lb, self.ub)

    # -------------------- main optimiser
    def optimize(self):
        P = np.random.uniform(self.lb, self.ub, (self.N, self.D))
        fit = np.array([self.f(x) for x in P])
        gbest = P[np.argmin(fit)].copy()
        history = [fit.min()]

        for _ in range(self.T):
            mean = P.mean(axis=0)

            # ---- 1. Select phase
            r = np.random.rand(self.N, 1)
            P = gbest + self.alpha * r * (mean - P)
            P = self._clip(P)

            # ---- 2. Spiral search phase
            theta = self.a_spiral * np.pi * np.random.rand(self.N, 1)
            rad = theta + self.R_spiral * np.random.rand(self.N, 1)
            xr = rad * np.sin(theta)
            yr = rad * np.cos(theta)
            rest = np.random.uniform(-1, 1, (self.N, self.D - 2))
            spiral_move = np.hstack([xr, yr, rest])
            P = mean + spiral_move
            P = self._clip(P)

            # ---- 3. Swoop phase
            xr_s = rad * np.sinh(theta)
            yr_s = rad * np.cosh(theta)
            x1 = xr_s / (np.abs(xr_s).max() + 1e-12)
            y1 = yr_s / (np.abs(yr_s).max() + 1e-12)
            x1 = np.repeat(x1, self.D, axis=1)
            y1 = np.repeat(y1, self.D, axis=1)
            P = (np.random.rand(self.N, 1) * gbest +
                 x1 * (P - self.c1 * mean) +
                 y1 * (P - self.c2 * gbest))
            P = self._clip(P)

            # -- evaluate
            fit = np.array([self.f(x) for x in P])
            if fit.min() < self.f(gbest):
                gbest = P[np.argmin(fit)].copy()
            history.append(self.f(gbest))
        return gbest, history[-1], history


# ------------------------------------------------------------
#  Benchmarklar
# ------------------------------------------------------------

def run_once(alg, name):
    start = time.time();  sol, best, hist = alg.optimize();  elapsed = time.time() - start
    return {"name": name, "best": best, "time": elapsed, "history": hist}


def test_algorithms(func_name, dim, lb, ub, iters=50, pop=30):
    f = getattr(TestFunctions, func_name)
    gsa = GoldenSineAlgorithm(f, dim, lb, ub, pop, iters)
    bea = BaldEagle(f, dim, lb, ub, pop, iters)
    res_gsa = run_once(gsa, "Golden Sine")
    res_bea = run_once(bea, "Bald Eagle")
    return {"function": func_name, "dimension": dim, "gsa": res_gsa, "bea": res_bea}


def compare_suite():
    suite = [
        ("sphere",     10, -10,   10),
        ("rosenbrock", 10, -5,    10),
        ("ackley",     10, -32.768, 32.768),
        ("rastrigin",  10, -5.12,  5.12),
        ("griewank",   10, -600,  600)
    ]
    return [test_algorithms(*case) for case in suite]


# ------------------------------------------------------------
#  Convergence & result plots
# ------------------------------------------------------------

def plot_convergence(results):
    plt.figure(figsize=(15, 10))
    for i, result in enumerate(results, 1):
        plt.subplot(3, 2, i)
        plt.plot(result['gsa']['history'], label='Golden Sine', color='gold')
        plt.plot(result['bea']['history'], label='Bald Eagle', color='blue')
        plt.title(f"{result['function'].capitalize()} Function")
        plt.xlabel('Iteration')
        plt.ylabel('Best fitness')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_3d_surface(func, bounds, title='Function'):
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = func(np.array([X[i,j], Y[i,j]]))
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(title)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


# ------------------------------------------------------------
#  Main 
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Running comparison suite…")
    
    plot_3d_surface(TestFunctions.rosenbrock, (-3,3), title='Rosenbrock')
    plot_3d_surface(TestFunctions.sphere, (-5,5), title='Sphere')
    plot_3d_surface(TestFunctions.ackley, (-5,5), title='Ackley')
    plot_3d_surface(TestFunctions.rastrigin, (-5,5), title='Rastrigin')
    plot_3d_surface(TestFunctions.griewank, (-5,5), title='Griewank')
    results = compare_suite()
    # Display brief table
    tbl = []
    for r in results:
        tbl.append({
            "Func": r["function"],
            "Best GSA": f"{r['gsa']['best']:.3e}",
            "Best BES": f"{r['bea']['best']:.3e}",
            "t_GSA(s)": f"{r['gsa']['time']:.2f}",
            "t_BES(s)": f"{r['bea']['time']:.2f}"
        })
    print(pd.DataFrame(tbl).to_string(index=False))

    # plot convergence
    plot_convergence(results)
    
    

