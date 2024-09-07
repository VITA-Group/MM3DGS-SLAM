import math

import numpy as np
import torch
import torch.nn.functional as F

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    "sxyz": (0, 0, 0, 0),
    "sxyx": (0, 0, 1, 0),
    "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0),
    "syzx": (1, 0, 0, 0),
    "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0),
    "syxy": (1, 1, 1, 0),
    "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0),
    "szyx": (2, 1, 0, 0),
    "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1),
    "rxyx": (0, 0, 1, 1),
    "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1),
    "rxzy": (1, 0, 0, 1),
    "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1),
    "ryxy": (1, 1, 1, 1),
    "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1),
    "rxyz": (2, 1, 0, 1),
    "rzyz": (2, 1, 1, 1),
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

G = torch.tensor([0.0, -9.80665, 0.0])  # Note that this is in camera optical frame


def euler_matrix(ai, aj, ak, axes="sxyz"):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    """

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    # M = numpy.identity(4)
    M = torch.eye(4).to(ai)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def preintegrate_imu(imu_meas_list, w2c, lin_vel, c2i, dt_imu):
    """
    Propagate camera pose based on IMU measurements

    Args:
        imu_meas_list (tensor): IMU measurements from last timestep to current timestep
        i2w (tensor): current tracked IMU pose. Modified
        lin_vel (tensor): current linear velocity. Modified
        dt_cam: time between camera frames
        dt_imu: time between imu measurements
    Returns:
        i2w: propagated pose
        lin_vel: propagated velocity
    """
    # Transform camera frame to IMU frame
    c2w = torch.inverse(w2c)
    i2w = c2w @ torch.inverse(c2i)

    # Then, do IMU preintegration
    for imu_meas in imu_meas_list:
        lin_accel = imu_meas[25:28]
        # lin_accel[1:] = 0
        # print(lin_accel)
        ang_vel = imu_meas[13:16]
        # print(ang_vel)

        # Remove the gravity component
        lin_accel -= i2w[:3, :3].T @ G.to(i2w)

        # Preintegrate
        change_in_position = lin_vel * dt_imu + 0.5 * lin_accel * dt_imu * dt_imu
        lin_vel += lin_accel * dt_imu
        change_in_orientation = ang_vel * dt_imu

        # Propagate pose
        delta = euler_matrix(*change_in_orientation, axes="sxyz")
        delta[0:3, 3] = change_in_position  # delta is i(now)->i(prev)
        i2w = i2w @ delta

    # Transform back to camera frame
    c2w = i2w @ c2i
    w2c = torch.inverse(c2w)

    return w2c, lin_vel


def propagate_imu(camm1, camm2, imu_meas_list, c2i, dt_cam, dt_imu):
    """
    Propagate camera pose based on IMU measurements

    Args:
        camm1 (tensor): pose at idx-1
        camm2 (tensor): pose at idx-2
        imu_meas_list (tensor): IMU measurements from last timestep to current timestep
        c2i (tensor): camera-to-imu homogeneous transformation matrix
        dt_cam: time between camera frames
        dt_imu: time between imu measurements
    Returns:
        cam (tensor): propagated pose
    """
    w2cm1 = get_camera_from_tensor(camm1)
    w2cm2 = get_camera_from_tensor(camm2)

    # Transform camera frame to IMU frame
    c2wm1 = torch.inverse(w2cm1)
    c2wm2 = torch.inverse(w2cm2)
    i2wm1 = c2wm1 @ torch.inverse(c2i)
    i2wm2 = c2wm2 @ torch.inverse(c2i)

    i2w = i2wm1.clone()  # Tracked IMU pose

    # Get the linear velocity using the constant velocity model
    rel_T = torch.inverse(i2wm2) @ i2wm1  # rel transform from i(now-1)->i(now-2)
    lin_vel = rel_T[:3, 3] / dt_cam

    # Then, do IMU preintegration
    for imu_meas in imu_meas_list:
        lin_accel = imu_meas[25:28]
        ang_vel = imu_meas[13:16]

        # Remove the gravity component
        lin_accel -= i2w[:3, :3].T @ G.to(i2w)

        # Preintegrate
        change_in_position = lin_vel * dt_imu + 0.5 * lin_accel * dt_imu * dt_imu
        change_in_orientation = ang_vel * dt_imu

        # Propagate pose
        delta = euler_matrix(*change_in_orientation, axes="sxyz")
        delta[0:3, 3] = change_in_position  # delta is i(now)->i(prev)
        i2w = i2w @ delta

        # TODO: perform covariance propagation

    # Transform back to camera frame
    c2w = i2w @ c2i
    w2c = torch.inverse(c2w)

    return get_tensor_from_camera(w2c)


def propagate_const_vel(camm1, camm2):
    """
    Propagate camera pose based on constant velocity model

    Args:
        camm1 (tensor): pose at idx-1
        camm2 (tensor): pose at idx-2
    Returns:
        cam (tensor): propagated pose
    """
    pre_w2c = get_camera_from_tensor(camm1)
    delta = pre_w2c @ get_camera_from_tensor(camm2).inverse()
    cam = get_tensor_from_camera(delta @ pre_w2c)
    return cam


def quadmultiply(q1, q2):
    """
    Multiply two quaternions together using quaternion arithmetic
    """
    # Extract scalar and vector parts of the quaternions
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    # Calculate the quaternion product
    result_quaternion = torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )

    return result_quaternion


def quad2rotation(q):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    if not isinstance(q, torch.Tensor):
        q = torch.tensor(q).cuda()

    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3)).to(q)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def rotation2quad(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix).cuda()

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def get_camera_from_tensor(inputs):
    """
    Convert quaternion and translation to transformation matrix.

    """
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.tensor(inputs).cuda()

    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)

    quad, T = inputs[:, :4], inputs[:, 4:]
    w2c = torch.eye(4).to(inputs).float()
    w2c[:3, :3] = quad2rotation(quad)
    w2c[:3, 3] = T
    return w2c


def get_tensor_from_camera(RT, Tquad=False):
    """
    Convert transformation matrix to quaternion and translation.

    """
    if not isinstance(RT, torch.Tensor):
        RT = torch.tensor(RT).cuda()

    rot = RT[:3, :3].unsqueeze(0).detach()
    quat = rotation2quad(rot).squeeze()
    tran = RT[:3, 3].detach()

    return torch.cat([quat, tran])
