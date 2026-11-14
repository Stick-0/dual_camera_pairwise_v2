"""Read-only numeric geometry: metres, Rodrigues radians, native pixels.

This module never imports the earlier extension, model, data annotation or runtime.
"""
import cv2
import numpy as np


def camera_arrays(camera):
    K = np.array(camera['K'], dtype=float, copy=True)
    D = np.array(camera.get('D', []), dtype=float, copy=True)
    if camera.get('pixel_space') == 'native_undistorted':
        if np.any(D):
            raise ValueError('Undistorted pixels require an effective camera with zero distortion')
    return K, D


def pose(x):
    x = np.asarray(x, float)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(x[:3])[0]
    T[:3, 3] = x[3:6]
    return T


def pose_vector(T):
    T = np.asarray(T, float)
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def inverse(T):
    T = np.asarray(T, float)
    result = np.eye(4)
    result[:3, :3] = T[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ T[:3, 3]
    return result


def transform(T, X):
    return np.asarray(X, float) @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def project(points, camera):
    """Finite fixed-size projection; callers MUST retain separate depth penalties."""
    points = np.asarray(points, float).reshape(-1, 3).copy()
    if not len(points):
        return np.empty((0, 2))
    points[:, 2] = np.maximum(points[:, 2], 1e-4)
    K, D = camera_arrays(camera)
    with np.errstate(over='ignore', invalid='ignore'):
        uv = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return np.nan_to_num(uv, nan=1e8, posinf=1e8, neginf=-1e8).clip(-1e8, 1e8)


def normalized(uv, camera):
    K, D = camera_arrays(camera)
    return cv2.undistortPoints(np.asarray(uv, float).reshape(-1, 1, 2), K, D).reshape(-1, 2)


def triangulate(uv1, uv2, G, cameras):
    a = normalized(uv1, cameras['camera1'])
    b = normalized(uv2, cameras['camera2'])
    H = cv2.triangulatePoints(np.eye(3, 4), np.asarray(G)[:3], a.T, b.T).T
    divisor = np.where(np.abs(H[:, 3]) > 1e-12, H[:, 3], np.nan)
    return H[:, :3] / divisor[:, None]


def parallax_degrees(P, G):
    centre2 = inverse(G)[:3, 3]
    a = np.asarray(P, float)
    b = a - centre2
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.sum(a * b, axis=1) / np.maximum(den, 1e-12)
    return np.rad2deg(np.arccos(np.clip(cos, -1, 1)))


def covariance_whiteners(covariance, floor_px=1.):
    C = np.array(covariance, dtype=float, copy=True)
    C = (C + C.swapaxes(-1, -2)) / 2
    values, vectors = np.linalg.eigh(C)
    regular = (vectors * np.maximum(values, floor_px ** 2)[..., None, :]) @ vectors.swapaxes(-1, -2)
    L = np.linalg.cholesky(regular)
    W = np.linalg.inv(L)
    return W, regular, values


def hull_fraction(points, camera):
    points = np.asarray(points, np.float32)
    if len(points) < 3:
        return 0.
    return float(cv2.contourArea(cv2.convexHull(points))) / (camera['width'] * camera['height'])


def huber_weights(blocks, delta):
    norms = np.linalg.norm(np.asarray(blocks), axis=-1)
    return np.minimum(1., delta / np.maximum(norms, 1e-12))


def huber_cost(blocks, delta):
    norms = np.linalg.norm(np.asarray(blocks), axis=-1)
    return float(np.sum(np.where(norms <= delta, .5 * norms ** 2, delta * (norms - .5 * delta))))


def pose_error(T, reference):
    return {'translation_m':float(np.linalg.norm(T[:3, 3] - reference[:3, 3])),
            'rotation_deg':float(np.linalg.norm(cv2.Rodrigues(T[:3, :3] @ reference[:3, :3].T)[0]) * 180 / np.pi)}


def essential_candidates(points1, points2, cameras, seed, pixel_threshold=2.):
    """The threshold is converted from pixels to normalized camera units once."""
    if len(points1) < 5:
        return [], {'status':'insufficient_essential_points', 'count':len(points1)}
    a, b = normalized(points1, cameras['camera1']), normalized(points2, cameras['camera2'])
    focal = np.mean([cameras[c]['K'][axis][axis] for c in ('camera1', 'camera2') for axis in (0, 1)])
    threshold = pixel_threshold / focal
    cv2.setRNGSeed(int(seed))
    E, mask = cv2.findEssentialMat(a, b, np.eye(3), method=cv2.RANSAC, prob=.999, threshold=threshold)
    diagnostics = {'ransac_threshold_px':pixel_threshold, 'ransac_threshold_normalized':threshold,
                   'inlier_count':int(mask.sum()) if mask is not None else 0, 'count':len(a)}
    if E is None:
        diagnostics['status'] = 'essential_estimation_failed'
        return [], diagnostics
    output = []
    for e in np.asarray(E).reshape(-1, 3, 3):
        # Four decompositions are retained. Their signs are selected by actual
        # positive depth and the same robot/RGB objective, never reference poses.
        R1, R2, d = cv2.decomposeEssentialMat(e)
        for R in (R1, R2):
            for sign in (1., -1.):
                G = np.eye(4); G[:3, :3] = R; G[:3, 3] = sign * d.ravel()
                P = triangulate(points1, points2, G, cameras)
                positive = np.isfinite(P).all(1) & (P[:, 2] > .001) & (transform(G, P)[:, 2] > .001)
                output.append({'R':R, 'direction':sign*d.ravel(), 'unit_positive_fraction':float(positive.mean())})
    output.sort(key=lambda v:-v['unit_positive_fraction'])
    diagnostics['status'] = 'essential_estimated'
    return output[:2], diagnostics
