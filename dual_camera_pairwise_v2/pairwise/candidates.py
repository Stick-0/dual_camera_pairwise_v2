"""Fixed original voting hypotheses, normalized support and Gaussian+uniform EM.

Repeated hypotheses do not become repeated keypoint measurements. Scores are
nonnegative original angular support counts (vote2.py), not calibrated probabilities.
"""
import numpy as np
from scipy.special import logsumexp


def prepare_candidates(observation, camera, config=None):
    options = dict(kernel_std_px=2., outlier_probability=.05, max_em_iterations=10,
                   oob_margin_px=0.)
    options.update(config or {})
    sigma = float(options['kernel_std_px']); epsilon = float(options['outlier_probability'])
    if sigma <= 0 or not 0 <= epsilon < 1:
        raise ValueError('Invalid candidate kernel or outlier probability')
    valid = np.asarray(observation['valid_keypoints'], bool)
    hypotheses = observation.get('hypotheses')
    if hypotheses is None or len(hypotheses) != len(valid):
        return None
    margin = float(options['oob_margin_px'])
    width, height = camera['width'], camera['height']
    area = (width + 2*margin) * (height + 2*margin)
    modes = []
    for k, h in enumerate(hypotheses):
        uv = np.array(h.get('uv', []), float, copy=True).reshape(-1, 2)
        scores = np.array(h.get('original_scores', []), float, copy=True)
        if len(uv) != len(scores):
            raise ValueError('Candidate coordinate/score count mismatch')
        if np.any(scores[np.isfinite(scores)] < 0):
            raise ValueError('Negative scores cannot be interpreted as vote support')
        good = np.isfinite(uv).all(1) & np.isfinite(scores) & (scores > 0)
        good &= (uv[:, 0] >= -margin) & (uv[:, 0] < width+margin) & (uv[:, 1] >= -margin) & (uv[:, 1] < height+margin)
        good &= valid[k]
        kept = np.flatnonzero(good)
        alpha = scores[kept].copy()
        if len(alpha):
            alpha /= alpha.sum()
        modes.append({'mu':uv[kept].copy(), 'alpha':alpha, 'original_indices':kept,
                      'valid':bool(len(kept)), 'original_count':len(uv)})
    return {'modes':modes, 'sigma_px':sigma, 'epsilon':epsilon,
            'uniform_density_px_inverse_squared':1./area,
            'domain':[-margin, -margin, width+margin, height+margin],
            'kernel_source':'explicit_isotropic_not_raw_voting_covariance',
            'outlier_interpretation':'uniform_over_retained_observation_pixel_domain_constant_in_projected_pose; approximate_contaminated_observation_likelihood_not_normalized_pose_posterior',
            'score_source':'normalized_original_nonnegative_angular_vote_support',
            'clustering':'none_original_hypotheses_kept', 'options':options}


def responsibilities(projected, prepared):
    output = []
    nll = 0.
    sigma, epsilon = prepared['sigma_px'], prepared['epsilon']
    log_normalizer = np.log(2*np.pi*sigma*sigma)
    log_out = np.log(epsilon*prepared['uniform_density_px_inverse_squared']) if epsilon else -np.inf
    for z, mode in zip(np.asarray(projected), prepared['modes']):
        if not mode['valid']:
            output.append({'responsibilities':np.zeros(0), 'outlier_responsibility':1.,
                           'entropy':0., 'selected_index':None, 'nll':0.})
            continue
        errors = (mode['mu']-z)/sigma
        logs = np.log1p(-epsilon)+np.log(mode['alpha'])-log_normalizer-.5*np.sum(errors**2, axis=1)
        all_logs = np.r_[logs, log_out]
        log_likelihood = float(logsumexp(all_logs))
        probs = np.exp(all_logs-log_likelihood)
        selected = int(np.argmax(probs[:-1]))
        entropy = float(-np.sum(probs[probs>0]*np.log(probs[probs>0])))
        output.append({'responsibilities':probs[:-1], 'outlier_responsibility':float(probs[-1]),
                       'entropy':entropy, 'selected_index':selected, 'nll':-log_likelihood})
        nll -= log_likelihood
    return output, float(nll)


def selected_candidates(prepared, posterior):
    uv = np.full((len(posterior), 2), np.nan)
    valid = np.zeros(len(posterior), bool)
    for k, (mode, post) in enumerate(zip(prepared['modes'], posterior)):
        h = post['selected_index']
        if h is not None and post['outlier_responsibility'] < .5:
            uv[k] = mode['mu'][h]
            valid[k] = True
    return uv, valid


def candidate_residuals(projected, prepared, posterior):
    rows = []
    for z, mode, post in zip(projected, prepared['modes'], posterior):
        if mode['valid']:
            rows.extend((np.sqrt(post['responsibilities'])[:, None] * (mode['mu']-z) / prepared['sigma_px']).tolist())
    return np.asarray(rows, float).reshape(-1, 2)


def candidate_seed_pixels(prepared):
    """Raw fixed hypotheses only; no synthetic/interpolated candidates are added."""
    n = len(prepared['modes'])
    ordered = [np.argsort(-m['alpha'], kind='stable') for m in prepared['modes']]
    seeds = []
    # Two coherent global assignments plus one changed mode at a time. This is a
    # deterministic initialization budget, shared without change by mono/dual.
    choices = [[0]*n, [1]*n] + [[int(i == k) for i in range(n)] for k in range(n)]
    for choice in choices:
        uv = np.full((n, 2), np.nan)
        for k, mode in enumerate(prepared['modes']):
            if mode['valid']:
                rank = min(choice[k], len(ordered[k])-1)
                uv[k] = mode['mu'][ordered[k][rank]]
        if not any(np.array_equal(uv, old, equal_nan=True) for old in seeds):
            seeds.append(uv)
    return seeds
