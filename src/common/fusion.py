"""Six-axis IMU orientation fusion with adaptive filter gain."""

import numpy as np

class IMUFusion:

    def __init__(
        self,
        initial_quaternion=None,
        gyro_bias=None,
        beta=0.05,
        delta_t=1 / 100,
        beta_mode='continuous',
        use_accel_weight=True
    ):
        self.q = np.array([1.0, 0.0, 0.0, 0.0]) if initial_quaternion is None else np.array(initial_quaternion)
        self.gyro_bias = np.zeros(3) if gyro_bias is None else np.array(gyro_bias)
        self.beta = beta
        self.delta_t = delta_t
        self.beta_mode = beta_mode
        self.use_accel_weight = use_accel_weight
        self.beta_static = 0.28
        self.beta_slow = 0.035
        self.beta_fast = 0.008
        self.th_static = 1.5
        self.th_fast = 4.5

    @staticmethod
    def normalize(v):
        return v if np.linalg.norm(v) == 0 else v / np.linalg.norm(v)

    def update(self, accel, gyro):
        accel_raw = np.array(accel, dtype=float)
        accel = self.normalize(accel)
        gyro_raw = np.array(gyro) - self.gyro_bias
        speed = np.linalg.norm(gyro_raw)
        if self.beta_mode == 'continuous':
            speed_clip = np.clip(speed, self.th_static, self.th_fast + 2.0)
            self.beta = self.beta_static - (speed_clip - self.th_static) * (self.beta_static - self.beta_fast) / (self.th_fast + 2.0 - self.th_static)
        elif speed < self.th_static:
            self.beta = self.beta_static
        elif speed < self.th_fast:
            self.beta = self.beta_slow
        else:
            self.beta = self.beta_fast
        if self.use_accel_weight:
            mag_err = abs(np.linalg.norm(accel_raw) - 1.0)
            w = max(0.0, 1.0 - 5.0 * mag_err)
            self.beta *= w
        gyro_dz = gyro_raw.copy()
        gyro_dz[np.abs(gyro_dz) < 2.0] = 0.0
        gyro_rps = np.radians(gyro_dz)
        q_dot = 0.5 * self.q_prod(self.q, [0.0, *gyro_rps])
        self.q = self.normalize(self.q + q_dot * self.delta_t)
        self.apply_accel_correction(accel)

    @staticmethod
    def q_prod(q, r):
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array(
            [
                -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                x1 * w0 - y1 * z0 + z1 * y0 + w1 * x0,
                x1 * z0 + y1 * w0 - z1 * x0 + w1 * y0,
                -x1 * y0 + y1 * x0 + z1 * w0 + w1 * z0
            ]
        )

    def rotate_vector(self, q, v):
        return self.q_prod(self.q_prod(q, [0.0, *v]), [q[0], -q[1], -q[2], -q[3]])[1:]

    def apply_accel_correction(self, accel):
        gravity = np.array([0.0, 0.0, 1.0])
        g_est = self.rotate_vector([self.q[0], -self.q[1], -self.q[2], -self.q[3]], gravity)
        err = np.cross(accel, g_est)
        self.q = self.normalize(self.q + np.array([0.0, *err]) * self.beta * self.delta_t)

    def get_quaternion(self):
        return self.q
