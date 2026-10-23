"""MSS tanker benchmark, converted from Fossen NED to local z-up axes.

Source tables use Re[H exp(+i omega t)]. Public open-water, zero-speed data;
not a calibrated berth or a maneuvering model. See reference provenance.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import lsim, StateSpace

PLANAR = [0, 1, 5]
SIGNS = np.array([1., -1., -1.])
MSS_COMMIT = "a175d836808ff723924198fc4fad352054aa18e2"


class TankerHydrodynamics:
    def __init__(self, directory="data/reference/mss_tanker"):
        directory = Path(directory)
        self.directory = directory
        self.vessel = loadmat(directory / "tanker.mat", simplify_cells=True)["vessel"]
        self.abc = loadmat(directory / "tankerABC.mat", simplify_cells=True)["vesselABC"]
        self.main = self.vessel["main"]
        self.omega = self.vessel["forceRAO"]["w"]
        headings = np.r_[self.vessel["headings"], 2*np.pi]
        rao = self.vessel["forceRAO"]
        table = np.stack([rao["amp"][i][:, :, 0]*np.exp(1j*rao["phase"][i][:, :, 0])
                          for i in PLANAR], axis=-1)
        table = np.concatenate((table, table[:, :1]), axis=1)
        self._interpolate = RegularGridInterpolator((self.omega, headings), table,
                                                    bounds_error=True)

    def force_rao(self, omega, direction_ccw):
        """Complex [N/m, N/m, Nm/m] for exp(-i omega t), relative TOWARD angle.

        Interpolate real/imaginary components with a periodic direction seam;
        reject frequency extrapolation. Conjugation changes time convention.
        """
        frequency, angle = np.broadcast_arrays(omega, direction_ccw)
        query = np.column_stack((frequency.ravel(), np.mod(-angle.ravel(), 2*np.pi)))
        return (self._interpolate(query).conj()*SIGNS).reshape(*frequency.shape, 3)

    def berth_force_raos(self, field, heading):
        """Incident and specularly reflected RAOs for every wave component.

        Global travel directions are theta (incident, toward the wall) and
        pi - theta (reflected); the ship sees them relative to its heading.
        Open-water RAOs ignore wall proximity and gap effects.
        """
        incident = self.force_rao(field.omega, field.direction - heading)
        reflected = self.force_rao(field.omega, np.pi - field.direction - heading)
        return incident, reflected

    def provenance(self):
        main = self.main
        return {"name": str(main["name"]), "length_m": float(main["Lpp"]), "beam_m": float(main["B"]),
                "draft_m": float(main["T"]), "mass_kg": float(main["m"]), "rho_kg_m3": float(main["rho"]),
                "sha256": {name: hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
                           for name in ("tanker.mat", "tankerABC.mat")},
                "source_commit": MSS_COMMIT, "time_convention": "Re[H exp(-i omega t)]",
                "frame": "z-up BODY, direction = wave travel TOWARD, CCW"}

    def planar_matrix(self, value):
        return np.asarray(value)[np.ix_(PLANAR, PLANAR)] * SIGNS[:, None]*SIGNS

    def response_system(self, stiffness):
        """Linearized 3-DOF Cummins model about a fixed heading/reference.

        M eta_ddot + Bv eta_dot + K eta + radiation = tau;
        z_dot = Ar z + Br eta_dot, radiation = Cr z + Dr eta_dot.
        Heave/roll/pitch constrained; Coriolis is second order and omitted.
        K is an explicitly assumed linear restraint, not a measured mooring.
        """
        mass = self.planar_matrix(self.abc["MRB"] + self.abc["MA"])
        # Restore reciprocity after small numerical asymmetry in WAMIT output.
        mass = (mass + mass.T)/2
        np.linalg.cholesky(mass)
        damping = self.planar_matrix(self.vessel["Bv"][:, :, 0])
        blocks = []
        size = 6
        for i, source_i in enumerate(PLANAR):
            for j, source_j in enumerate(PLANAR):
                ar = np.asarray(self.abc["Ar"][source_i, source_j])
                if ar.size:
                    br = np.asarray(self.abc["Br"][source_i, source_j]).reshape(-1)
                    cr = np.asarray(self.abc["Cr"][source_i, source_j]).reshape(-1)
                    dr = float(self.abc["Dr"][source_i, source_j])
                    blocks.append((i, j, slice(size, size+len(br)), ar, br, cr))
                    size += len(br)
                    damping[i, j] += SIGNS[i]*SIGNS[j]*dr
        inverse = np.linalg.inv(mass)
        a = np.zeros((size, size)); b = np.zeros((size, 3))
        a[:3, 3:6] = np.eye(3)
        a[3:6, :3] = -inverse @ np.diag(stiffness)
        a[3:6, 3:6] = -inverse @ damping
        b[3:6] = inverse
        for i, j, section, ar, br, cr in blocks:
            a[section, section] = ar
            a[section, 3+j] = br*SIGNS[j]
            a[3:6, section] -= inverse[:, i, None] * (cr*SIGNS[i])[None, :]
        if np.max(np.linalg.eigvals(a).real) >= 0:
            raise ValueError("Unstable restrained radiation model")
        c = np.zeros((6, size)); c[:, :6] = np.eye(6)
        return StateSpace(a, b, c, np.zeros((6, 3)))

    def response(self, time, loads, stiffness):
        """Zero initial state, linearly interpolated loads, local displacement/rate."""
        _, states, _ = lsim(self.response_system(stiffness), U=loads, T=time)
        return states
