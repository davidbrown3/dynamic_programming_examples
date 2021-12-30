import json
import jax.numpy as np

from ddp_examples.cartpole import CartPole
from ddpy.solvers import DifferentialDynamicProgramming
from ddpy.solvers import quadratic_cost_function

debug = False

time_total = 10
dt = 1e-2
N_Steps = time_total / dt

plant = CartPole(dt, 0)

g_xx = np.diag(np.array([0.0, 0.0, 0.0, 0.0]))
g_uu = np.array([[1e-7]])
g_xu = np.zeros([4, 1])
g_ux = np.zeros([1, 4])
g_x = np.array([[0.0, 0.0, 0.0, 0.0]])
g_u = np.array([[0.0]])
running_cost_function = quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_ux=g_ux, g_uu=g_uu, g_x=g_x, g_u=g_u)

g_xx = np.diag(np.array([1.0, 0.0, 1.0, 1.0]))
g_uu = np.array([[0.0]])
g_xu = np.zeros([4, 1])
g_ux = np.zeros([1, 4])
g_x = np.array([[0.0, 0.0, 0.0, 0.0]])
g_u = np.array([[0.0]])
terminal_cost_function = quadratic_cost_function(g_xx=g_xx, g_xu=g_xu, g_ux=g_ux, g_uu=g_uu, g_x=g_x, g_u=g_u)

solver = DifferentialDynamicProgramming(plant=plant,
                                        running_cost_function=running_cost_function,
                                        terminal_cost_function=terminal_cost_function,
                                        debug=debug,
                                        order=2
                                        )

states_initial = np.array([[5.0], [0.0], [np.deg2rad(180)], [0.0]])

xs, us, costs = solver.solve(
    states_initial,
    time_total,
    controls_initial=plant.random_control_law,
    iterations=400
)

with open('cartpole.json', 'w') as f:
    json.dump(
        dict(
            position=[float(x) for x in xs[:, 0]],
            angle=[float(x) for x in xs[:, 2]]
        ), f)
