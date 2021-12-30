from collections import namedtuple

import jax.numpy as np
import pytest
from ddp_examples.cartpole import CartPole


@pytest.fixture(scope="session")
def plant():
    return CartPole(dt=0.1)


angular_positions = np.logspace(0, 7, num=5)
angular_velocitys = np.logspace(0, 7, num=5)
velocitys = np.logspace(0, 7, num=5)
forces = np.logspace(0, 7, num=5)

Case = namedtuple('Case', 'angular_position angular_velocity velocity force')

cases = []
for angular_position in angular_positions:
    for angular_velocity in angular_velocitys:
        for velocity in velocitys:
            for force in forces:
                cases.append(Case(angular_position, angular_velocity, velocity, force))


@pytest.mark.parametrize("angular_position,angular_velocity,velocity,force", cases)
def test_plant(plant, angular_position, angular_velocity, velocity, force):

    position = 0.0
    x = np.array([[position], [velocity], [angular_position], [angular_velocity]])
    u = np.array([[force]])

    xd = plant._derivatives(x, u)
    assert not np.any(np.isnan(xd))

    xnew = plant.step(x, u)
    assert not np.any(np.isnan(xnew))
