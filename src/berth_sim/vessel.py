"""Low-frequency 3-DOF Fossen-form demonstration, zero current.

    Frame: x points toward wall, y lies along wall, heading CCW, z up.
    All coefficients must follow this convention (not blindly copied NED).
    Constant added mass/damping is not a full radiation-memory model.
"""
import numpy as np
from .wind import rotation, wind_load


def coriolis(mass_matrix, velocity):
    """Skew-symmetric planar Coriolis representation for constant inertia."""
    momentum = mass_matrix @ velocity
    return np.array([[0., 0., -momentum[1]],
                     [0., 0., momentum[0]],
                     [momentum[1], -momentum[0], 0.]])


def rhs(state, wind, wave_force, config):
    eta, velocity = state[:3], state[3:]
    mass = np.diag(config["mass_diagonal"])
    damping = np.asarray(config["linear_damping"]) + np.asarray(config["quadratic_damping"]) * np.abs(velocity)
    error = eta - np.asarray(config["hold_position"])
    error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))
    # Position control is calculated in BODY coordinates, actuator limited.
    error[:2] = rotation(eta[2]).T @ error[:2]
    control = -np.asarray(config["hold_kp"]) * error - np.asarray(config["hold_kd"]) * velocity
    limit = np.asarray(config["control_limits"])
    control = np.clip(control, -limit, limit)
    load = control + wind_load(wind, eta[2], velocity, config["wind_load"]) + wave_force
    acceleration = np.linalg.solve(mass, load - coriolis(mass, velocity) @ velocity - damping * velocity)
    return np.r_[rotation(eta[2]) @ velocity[:2], velocity[2], acceleration]


def simulate_vessel(time, wind, wave_force, config):
    """RK4 with interpolated loads; no contact/fender model is included."""
    state = np.empty((len(time), 6))
    state[0] = config["initial_state"]
    for i, dt in enumerate(np.diff(time)):
        wm = (wind[i] + wind[i + 1]) / 2
        fm = (wave_force[i] + wave_force[i + 1]) / 2
        k1 = rhs(state[i], wind[i], wave_force[i], config)
        k2 = rhs(state[i] + dt * k1 / 2, wm, fm, config)
        k3 = rhs(state[i] + dt * k2 / 2, wm, fm, config)
        k4 = rhs(state[i] + dt * k3, wind[i + 1], wave_force[i + 1], config)
        state[i + 1] = state[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        if not np.isfinite(state[i + 1]).all():
            raise RuntimeError("Non-finite vessel state: reduce dt or revise coefficients")
    return state


def first_order_force(time, field, incident_rao, reflected_rao, reference_x, envelope):
    """Complex force RAOs [N/m, N/m, Nm/m], shape (frequency, 3).

    Fixed heading/reference point approximation. Incident/reflected RAOs
    must correspond to their respective directions. Not for maneuvering
    over large distances or using open-water RAOs in a narrow gap.
    """
    expected = (len(field.omega), 3)
    if np.shape(incident_rao) != expected or np.shape(reflected_rao) != expected:
        raise ValueError(f"Force RAOs must have shape {expected}")
    coefficients = field.amplitude[:, None] * (
        np.asarray(incident_rao) * np.exp(1j * field.kx * reference_x)[:, None]
        + np.asarray(reflected_rao) * (field.reflection * np.exp(1j * (-field.kx * reference_x + field.reflection_phase)))[:, None])
    result = np.empty((len(time), 3))
    for start in range(0, len(time), 2048):
        stop = min(start + 2048, len(time))
        phasor = np.exp(1j * (field.phase - time[start:stop, None] * field.omega))
        result[start:stop] = (phasor @ coefficients).real
    return result * envelope[:, None]
