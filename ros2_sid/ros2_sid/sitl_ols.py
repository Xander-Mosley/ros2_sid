"""Fit the stock SIM_Plane aerodynamic equations from recorded SITL flights.

Uses truth attitude/velocity for alpha/beta, IMU rates/accelerations for outputs,
and recorded PWM passed through the simulator's servo model. No aerodynamic
reference coefficient enters the regression. Geometry, mass, thrust calibration,
CG offset and servo dynamics are assumed known. Only conventional plane supported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp, RotationSpline

from ros2_sid.realtime_ols_utils import ordinary_least_squares


def read_log(path):
    """Read relevant DataFlash messages, retaining the primary IMU only."""
    from pymavlink import mavutil

    log = mavutil.mavlink_connection(str(path))
    data = {k: [] for k in ('SIM', 'SIM2', 'IMU', 'RCOU', 'SXC', 'SXO')}
    params = {}
    while (msg := log.recv_match()) is not None:
        kind = msg.get_type()
        if kind == 'PARM':
            params[msg.Name] = msg.Value
        if kind in data and (kind != 'IMU' or msg.I == 0):
            data[kind].append(msg.to_dict())
    log.close()
    return data, params


def interpolate(rows, fields, t):
    """Interpolate timestamped scalar columns onto a common boot-time grid."""
    times = np.array([r['TimeUS'] * 1e-6 for r in rows])
    return np.column_stack([np.interp(t, times, [r[k] for r in rows]) for k in fields])


def servo_inputs(rows, t, tau):
    """Reconstruct the 1-pole SIM_SERVO_SPEED filter at simulator resolution."""
    ts = np.array([r['TimeUS'] * 1e-6 for r in rows])
    pwm = np.array([[r[f'C{i}'] for i in (1, 2, 4, 3)] for r in rows])
    raw = (pwm - [1500, 1500, 1500, 1000]) / [500, 500, 500, 1000]
    fine = np.arange(ts[0], t[-1] + .001, .001)
    held = raw[np.maximum(0, np.searchsorted(ts, fine, side='right') - 1)]
    filtered = np.empty_like(held)
    filtered[0] = held[0]
    gain = 1.0 if tau <= 0 else .001 / (tau + .001)
    for i in range(1, len(fine)):
        filtered[i] = filtered[i-1] + gain * (held[i] - filtered[i-1])
    return np.column_stack([np.interp(t, fine, filtered[:, i]) for i in range(4)])


def flight_arrays(data, params, state_source="imu"):
    """Construct dimensionless regressors and force/moment observations."""
    for name in ('SIM', 'SIM2', 'IMU', 'RCOU'):
        if len(data[name]) < 20:
            raise ValueError(f'Insufficient {name} records')
    for name in ('SIM_WIND_SPD', 'SIM_SERVO_DELAY', 'SIM_SERVO_FILTER'):
        if params.get(name, 0) != 0:
            raise ValueError(f'{name} must be zero for this benchmark')
    start = max(data[k][0]['TimeUS'] for k in ('SIM','SIM2','IMU','RCOU')) * 1e-6 + 1
    end = min(data[k][-1]['TimeUS'] for k in ('SIM','SIM2','IMU','RCOU')) * 1e-6 - 1
    dt = .02
    t = np.arange(start, end, dt)
    sim = data['SIM']
    times = np.array([r['TimeUS'] * 1e-6 for r in sim])
    quat = np.array([[r[k] for k in ('Q2','Q3','Q4','Q1')] for r in sim])
    rot = Slerp(times, Rotation.from_quat(quat))(t)
    velocity = interpolate(data['SIM2'], ('VN','VE','VD'), t)
    vb = rot.inv().apply(velocity)  # zero-wind run, NED -> FRD
    speed = np.linalg.norm(vb, axis=1)
    alpha = np.arctan2(vb[:, 2], vb[:, 0])
    beta = np.arctan2(vb[:, 1], vb[:, 0])  # SIM_Plane uses atan2, not asin
    alt = interpolate(sim, ('Alt',), t)[:, 0]
    h = 6356766 * alt / (6356766 + alt)
    rho = 1.225 * (1 - .0065 * h / 288.15) ** (9.80665 / (.0065 * 287.053072) - 1)
    # SIM_Aircraft::update_eas_airspeed stores EAS in `airspeed`, which
    # SIM_Plane uses together with local density. Reproduce that convention.
    speed *= np.sqrt(rho / 1.225)
    qs = .5 * rho * speed**2 * .45
    imu = interpolate(data['IMU'], ('GyrX','GyrY','GyrZ','AccX','AccY','AccZ'), t)
    if state_source == "truth":
        spline = RotationSpline(times, Rotation.from_quat(quat))
        imu[:, :3] = spline(t, 1)
        inertial_accel = savgol_filter(velocity, 7, 3, deriv=1, delta=dt, axis=0)
        imu[:, 3:6] = rot.inv().apply(inertial_accel - [0, 0, 9.80665])
    rates = imu[:, :3]
    # A short symmetric polynomial window reduces numerical differentiation noise.
    angular_accel = np.gradient(rates, dt, axis=0)
    # SXO captures surfaces at 50 Hz, versus the ordinary 25 Hz RCOU log.
    # Retain RCOU for throttle, which is absent from the surface output stream.
    servo_rows = data['RCOU']
    if data['SXO']:
        events = [(r['TimeUS'], {'C3': r['C3']}) for r in data['RCOU']]
        events += [(r['TimeUS'], {f"C{r['Ch']+1}": r['PWM']})
                   for r in data['SXO'] if r['Ch'] in (0, 1, 3)]
        values = dict(C1=1500, C2=1500, C3=1000, C4=1500)
        servo_rows = []
        for stamp, update in sorted(events, key=lambda item: item[0]):
            values.update(update)
            servo_rows.append(dict(TimeUS=stamp, **values))
    controls = servo_inputs(servo_rows, t, params.get('SIM_SERVO_SPEED', .14))
    force = 2 * imu[:, 3:6]
    force[:, 0] -= controls[:, 3] * (2 * 9.80665 / .7)
    cg_moment = np.cross(np.array([-.15, 0, -.05]), force)
    safe_v = np.maximum(speed, 1)
    safe_qs = np.maximum(qs, 1)
    phat = rates[:, 0] * 1.88 / (2 * safe_v)
    qhat = rates[:, 1] * .24 / (2 * safe_v)
    rhat = rates[:, 2] * 1.88 / (2 * safe_v)
    one = np.ones_like(t)
    lat = np.column_stack((one, beta, phat, rhat, controls[:, 0], controls[:, 2]))
    lon = np.column_stack((one, alpha, qhat, controls[:, 1]))
    moments = (angular_accel - cg_moment) / (safe_qs[:, None] * [1.88, .24, 1.88])
    lift = (force[:, 0] * np.sin(alpha) - force[:, 2] * np.cos(alpha)) / safe_qs
    drag = -(force[:, 0] * np.cos(alpha) + force[:, 2] * np.sin(alpha)) / safe_qs
    models = {
        'roll': (lat, moments[:, 0], ['c_l_0','c_l_b','c_l_p','c_l_r','c_l_deltaa','c_l_deltar']),
        'pitch': (lon, moments[:, 1], ['c_m_0','c_m_a','c_m_q','c_m_deltae']),
        'yaw': (lat, moments[:, 2], ['c_n_0','c_n_b','c_n_p','c_n_r','c_n_deltaa','c_n_deltar']),
        'sideforce': (lat, force[:, 1] / safe_qs, ['c_y_0','c_y_b','c_y_p','c_y_r','c_y_deltaa','c_y_deltar']),
        'lift': (lon, lift, ['c_lift_0','c_lift_a','c_lift_q','c_lift_deltae']),
        'drag': (np.column_stack((one, alpha, alpha**2, qhat, np.abs(controls[:, 1]))), drag,
                 ['drag_constant','drag_alpha','drag_alpha2','c_drag_q','c_drag_deltae']),
    }
    # Apply identical smoothing to outputs and every regressor, preserving OLS.
    models = {name: (savgol_filter(x, 9, 3, axis=0),
                     savgol_filter(y, 9, 3), names)
              for name, (x, y, names) in models.items()}
    valid = (speed > 12) & (alt > sim[0]['Alt'] + 20) & (np.abs(alpha) < .20) & (np.abs(beta) < .25)
    return t, models, valid


def maneuver_mask(run, data, t, prefix):
    """Match command sequence IDs to firmware receipt timestamps, excluding edges."""
    mask = np.zeros(len(t), dtype=bool)
    intervals = []
    for path in sorted(run.glob(f'{prefix}_*-commands.csv')):
        with path.open() as stream:
            ids = {int(r['sequence']) for r in csv.DictReader(stream) if int(r['enable'])}
        stamps = [r['TimeUS'] * 1e-6 for r in data['SXC'] if r['Seq'] in ids and r.get('Valid', 1)]
        if not stamps:
            raise ValueError(f'No firmware receipts matching {path}')
        lo, hi = min(stamps) + .3, max(stamps) - .3
        mask |= (t >= lo) & (t <= hi)
        intervals.append({'name': path.stem, 'start_s': lo, 'end_s': hi})
    if len(intervals) != 3:
        raise ValueError(f'Expected three {prefix} maneuvers, found {len(intervals)}')
    return mask, intervals


def fit_models(models, train, test):
    """Fit scaled OLS without reference values; evaluate disjoint maneuvers."""
    results = {}
    predictions = {}
    for name, (x, y, names) in models.items():
        scale = np.sqrt(np.mean(x[train]**2, axis=0))
        scale = np.where(scale > 0, scale, 1)
        scaled = x[train] / scale
        rank = np.linalg.matrix_rank(scaled)
        if rank != x.shape[1]:
            raise ValueError(f'{name} is rank deficient: {rank}/{x.shape[1]}')
        coef = ordinary_least_squares(y[train], scaled) / scale
        pred = x @ coef
        predictions[name] = pred
        def metrics(mask):
            res = y[mask] - pred[mask]
            return {'rmse': float(np.sqrt(np.mean(res**2))),
                    'r2': float(1 - np.sum(res**2) / np.sum((y[mask]-np.mean(y[mask]))**2))}
        results[name] = {'coefficients': dict(zip(names, map(float, coef))),
                         'rank': int(rank), 'scaled_condition': float(np.linalg.cond(scaled)),
                         'train': metrics(train), 'validation': metrics(test)}
    return results, predictions


def main():
    """Write coefficient comparison, validation plots and reproducible fit inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state-source', choices=('imu', 'truth'), default='imu')
    parser.add_argument('--include-transients', action='store_true',
                        help='Also train on airborne climb/settling, excluding entire validation maneuvers')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='SIM_Plane.h; comparison only')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execution = json.loads((args.run / 'execution.json').read_text())
    options = json.loads((args.run / 'test_options.json').read_text())
    if options['airframe'] != 'conventional':
        raise ValueError('Only the conventional plane model is supported')
    required = ['bag_validation'] + [f'{signal}_{axis}' for signal in ('chirp', 'multisine')
                                    for axis in ('roll', 'pitch', 'yaw')]
    if not all(execution.get(key) is True for key in required):
        raise ValueError('Requires a complete, successfully validated six-maneuver flight')
    data, params = read_log(args.run / 'logs/00000001.BIN')
    t, models, valid = flight_arrays(data, params, args.state_source)
    train, train_intervals = maneuver_mask(args.run, data, t, 'chirp')
    test, test_intervals = maneuver_mask(args.run, data, t, 'multisine')
    if args.include_transients:
        # Broaden alpha/speed coverage without fitting on held-out multisines.
        # Restore maneuver edges plus an additional filter guard around each.
        train = t < max(row['end_s'] for row in test_intervals)
        for row in test_intervals:
            train &= (t < row['start_s'] - .6) | (t > row['end_s'] + .6)
    train &= valid
    test &= valid
    if np.any(train & test) or min(train.sum(), test.sum()) < 100:
        raise ValueError('Insufficient or overlapping train/validation data')
    results, predictions = fit_models(models, train, test)
    for name, (_, y, _) in models.items():
        per_maneuver = {}
        for interval in test_intervals:
            mask = test & (t >= interval['start_s']) & (t <= interval['end_s'])
            residual = y[mask] - predictions[name][mask]
            total = np.sum((y[mask] - np.mean(y[mask]))**2)
            per_maneuver[interval['name']] = {
                'samples': int(mask.sum()),
                'rmse': float(np.sqrt(np.mean(residual**2))),
                'r2': float(1 - np.sum(residual**2) / total) if total > 0 else None,
            }
        results[name]['validation_by_maneuver'] = per_maneuver
    # Reference aerodynamic values are read only AFTER fitting.
    reference = {k: float(v) for k,v in re.findall(r'float\s+(\w+)\s*=\s*([-+\d.]+)\s*;', args.reference.read_text())}
    drag_denominator = np.pi * reference['oswald'] * reference['b']**2 / reference['s']
    reference.update(
        drag_constant=reference['c_drag_p'] + reference['c_lift_0']**2 / drag_denominator,
        drag_alpha=2*reference['c_lift_0']*reference['c_lift_a'] / drag_denominator,
        drag_alpha2=reference['c_lift_a']**2 / drag_denominator,
    )
    rows = []
    for model, result in results.items():
        for name, value in result['coefficients'].items():
            ref = reference.get(name)
            rows.append({'model': model, 'coefficient': name, 'estimate': value, 'reference': ref,
                         'absolute_error': None if ref is None else abs(value-ref),
                         'relative_error_percent': None if ref is None or ref == 0 else 100*abs((value-ref)/ref)})
    report = {'method': 'OLS; truth alpha/beta, selectable IMU/truth outputs, filtered PWM; known geometry/mass/CG/thrust',
              'reference_file': str(args.reference),
              'known_calibration': {'mass_kg': 2., 'area_m2': .45, 'span_m': 1.88,
                                    'chord_m': .24, 'cg_offset_m': [-.15, 0, -.05],
                                    'thrust_scale_n': 2*9.80665/.7,
                                    'surface_pwm_center': 1500, 'surface_pwm_half_range': 500,
                                    'force_airspeed': 'EAS = norm(velocity_air) * sqrt(rho/1.225)'},
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
              'dataflash_file': str(args.run / 'logs/00000001.BIN'),
              'dataflash_sha256': hashlib.sha256((args.run/'logs/00000001.BIN').read_bytes()).hexdigest(),
              'limitations': ['Not sensor-only: alpha and beta use simulator truth and zero wind.',
                              'One-step coefficient prediction, not free-running trajectory validation.',
                              'Correlated closed-loop regressors and sensor/servo timing can bias coefficients.',
                              'Drag is fitted as a polynomial in alpha; reference polynomial is derived from SIM_Plane.h.',
                              'No confidence intervals: residuals are temporally correlated.'],
              'state_source': args.state_source, 'train_samples': int(train.sum()), 'validation_samples': int(test.sum()),
              'include_airborne_transients': args.include_transients,
              'train_intervals': train_intervals, 'validation_intervals': test_intervals,
              'models': results, 'comparison': rows, 'servo_tau_s': params.get('SIM_SERVO_SPEED', .14)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'results.json').write_text(json.dumps(report, indent=2) + '\n')
    with (args.output/'coefficients.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    arrays = {'time_s': t, 'train': train, 'validation': test}
    for name, (x,y,_) in models.items():
        arrays[name+'_X'] = x
        arrays[name+'_y'] = y
    np.savez_compressed(args.output/'regression_data.npz', **arrays)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for ax, (name, (_, y, _)) in zip(axes.flat, models.items()):
        ax.plot(t[test], y[test], '.', ms=2, label='recorded')
        ax.plot(t[test], predictions[name][test], '.', ms=2, label='OLS prediction')
        ax.set_title(f"{name}: held-out R²={results[name]['validation']['r2']:.4f}")
        ax.set_xlabel('Boot time (s)')
        ax.set_ylabel('Dimensionless coefficient')
        ax.grid(alpha=.3)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(args.output/'validation.png', dpi=160)
    fig.savefig(args.output/'validation.pdf')
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
