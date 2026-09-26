"""Metrics for non-negative, bi-objective minimisation (minutes)."""
import math
import warnings


def deduplicate_front(front, digits=6):
    """Objective-space reporting only; retain original objective values."""
    seen, result = set(), []
    for p in sorted(front):
        key = tuple(round(float(v), digits) for v in p)
        if key not in seen:
            seen.add(key)
            result.append(tuple(p))
    return result


def nondominated_points(front):
    points = sorted(set(tuple(map(float, p)) for p in front))
    if any(len(p) != 2 or not all(math.isfinite(x) and x >= 0 for x in p)
           for p in points):
        raise ValueError("Expected finite, non-negative (f1, f2) pairs")
    result, best = [], math.inf
    for point in points:
        if point[1] < best:
            result.append(point)
            best = point[1]
    return result


def compute_nadir_reference(all_runs_pareto_fronts, margin=0.1):
    """Empirical union-front nadir + margin, NOT a proven true nadir.

    Pool only the same instance and objective definitions, across all methods
    and seeds. Freeze once; never estimate a new reference for each generation.
    A positive absolute margin also handles singleton/zero-wait fronts.
    """
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be positive and finite")
    points = nondominated_points(p for front in all_runs_pareto_fronts for p in front)
    if not points:
        raise ValueError("Cannot estimate a reference from empty fronts")
    nadir = tuple(max(p[j] for p in points) for j in range(2))
    return tuple(x + margin * max(x, 1.0) for x in nadir)


def hypervolume_2d(front, ref):
    """Area dominated inside [0, ref]; points outside the box contribute zero."""
    r1, r2 = map(float, ref)
    if not all(math.isfinite(x) and x > 0 for x in (r1, r2)):
        raise ValueError("HV reference coordinates must be finite and positive")
    points = nondominated_points(front)
    points = [p for p in points if p[0] <= r1 and p[1] <= r2]
    return sum(((points[i + 1][0] if i + 1 < len(points) else r1) - x)
               * (r2 - y) for i, (x, y) in enumerate(points))


def hv_fixed_ref_norm(front, ref):
    """HV/(ref_f1*ref_f2), with fixed physical lower bound (0, 0).

    WARNING: old running-ideal HV time series are NOT comparable. Recompute
    from saved fronts or rerun. Fixed scaling alone does not guarantee a
    monotone curve: finite-population selection may discard useful points.
    """
    return hypervolume_2d(front, ref) / (float(ref[0]) * float(ref[1]))


def hv_instance_norm(front, ref):
    """Deprecated compatibility entry point; never uses a running ideal."""
    warnings.warn("Use hv_fixed_ref_norm(front, ref)", DeprecationWarning,
                  stacklevel=2)
    return hv_fixed_ref_norm(front, ref)


def compute_true_extremes(reference_front):
    """Return actual endpoint PAIRS, not the unattainable componentwise ideal.

    Supply verified MIP endpoint solutions or a best-known reference front.
    The latter is an empirical estimate, not proof of the true Pareto boundary.
    """
    points = nondominated_points(reference_front)
    if not points:
        raise ValueError("Reference front is empty")
    return points[0], points[-1]


def _spread(front, extremes):
    points = nondominated_points(front)
    if len(points) < 2:
        return math.nan  # Undefined, not a misleading perfect score of zero.
    distances = [math.dist(a, b) for a, b in zip(points, points[1:])]
    mean = sum(distances) / len(distances)
    extent = (0.0 if extremes is None else
              math.dist(points[0], extremes[0]) + math.dist(points[-1], extremes[1]))
    denominator = extent + sum(distances)
    return ((extent + sum(abs(d - mean) for d in distances)) / denominator
            if denominator else math.nan)


def compute_spacing_metric(front):
    """Legacy adjacent-Euclidean spread with d_f=d_l=0 (dimensionless).

    NOT Schott's nearest-neighbour spacing: retain the requested old formula
    but label its exact definition. Distances use raw minutes on both axes.
    """
    return _spread(front, None)


def compute_deb_delta(front, true_extremes=None):
    """Deb spread with REQUIRED endpoint pairs; missing endpoints raise ValueError."""
    if true_extremes is None:
        raise ValueError("true_extremes is required for Deb_Delta")
    if len(true_extremes) != 2:
        raise ValueError("Supply two endpoint pairs")
    points = nondominated_points(true_extremes)
    if not points:
        raise ValueError("Invalid endpoints")
    if len(points) == 1 and tuple(true_extremes[0]) != tuple(true_extremes[1]):
        raise ValueError("Distinct endpoints must be mutually nondominated")
    return _spread(front, (points[0], points[-1]))


def diversity_metrics(front, true_extremes=None):
    result = {"Spacing": compute_spacing_metric(front)}
    if true_extremes is not None:
        result["Deb_Delta"] = compute_deb_delta(front, true_extremes)
    return result
