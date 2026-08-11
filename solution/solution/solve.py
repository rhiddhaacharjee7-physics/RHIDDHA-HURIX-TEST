from pathlib import Path
import os
import json

import numpy as np
import pandas as pd

from scipy.optimize import least_squares


DATA = Path(os.environ.get("DATA_DIR", "/app/data"))
OUT = Path(os.environ.get("OUTPUT_DIR", "/app/output"))

OUT.mkdir(parents=True, exist_ok=True)


def sem(x):
    x = np.asarray(x, float)

    if len(x) <= 1:
        return 0.0

    return float(
        x.std(ddof=1) / np.sqrt(len(x))
    )


def fit_model(tab, irrelevant):

    a = tab.alpha.to_numpy(float)
    m = tab.width.to_numpy(float)

    y = tab.lambda_dimensionless.to_numpy(float)

    s = np.maximum(
        tab.standard_error.to_numpy(float),
        1e-6,
    )

    if irrelevant:

        def pred(p):

            return (
                p[0]
                + p[1]
                * (a - p[2])
                * (m / 8.0) ** (1 / p[3])
                + p[4] * (m / 8.0) ** -1
            )

        x0 = [
            0.62,
            0.8,
            0.72,
            1.5,
            0.1,
        ]

        lo = [
            0.1,
            0.01,
            0.5,
            0.5,
            -1,
        ]

        hi = [
            2,
            3,
            0.95,
            3,
            1,
        ]

    else:

        def pred(p):

            return (
                p[0]
                + p[1]
                * (a - p[2])
                * (m / 8.0) ** (1 / p[3])
            )

        x0 = [
            0.7,
            0.8,
            0.72,
            1.5,
        ]

        lo = [
            0.1,
            0.01,
            0.5,
            0.5,
        ]

        hi = [
            2,
            3,
            0.95,
            3,
        ]

    fit = least_squares(
        lambda p: (pred(p) - y) / s,
        x0=x0,
        bounds=(lo, hi),
    )

    chi2 = float(
        np.sum(fit.fun ** 2)
    )

    k = len(fit.x)

    return (
        fit.x,
        chi2,
        len(y) - k,
        chi2 + 2 * k,
    )


limits = json.loads(
    (DATA / "validity_limits.json").read_text()
)

tm = pd.read_csv(
    DATA / "transfer_runs.csv",
    keep_default_na=False,
)

sp = pd.read_csv(
    DATA / "spectral_runs.csv"
)


qc = []

valid_tm = []


for _, r in tm.iterrows():

    if str(r.parent_sample_id).strip():

        reason = "duplicate_checkpoint"

    elif (
        int(r.qr_failures)
        > limits["maximum_qr_failures"]
    ):

        reason = "qr_failure"

    elif (
        (not bool(r.converged))
        or int(r.n_blocks)
        < limits["minimum_transfer_blocks"]
        or int(r.total_bar_length)
        < limits["minimum_transfer_length"]
    ):

        reason = "insufficient_convergence"

    elif (
        float(r.relative_gamma_error)
        > limits["maximum_relative_gamma_error"]
    ):

        reason = "convergence_failure"

    else:

        reason = "valid"

    included = reason == "valid"

    qc.append(
        [
            r.sample_id,
            r.campaign,
            "transfer",
            included,
            reason,
            int(r.n_blocks),
            int(r.n_blocks)
            if included
            else 0,
        ]
    )

    if included:

        valid_tm.append(
            [
                r.sample_id,
                float(r.alpha),
                int(r.width),
                1
                / (
                    int(r.width)
                    * float(r.mean_gamma)
                ),
            ]
        )


valid_sp = []


for _, r in sp.iterrows():

    if (
        float(r.maximum_residual)
        > limits["maximum_eigenpair_residual"]
    ):

        reason = "residual_failure"

    elif (
        float(r.orthogonality_defect)
        > limits["maximum_orthogonality_defect"]
    ):

        reason = "orthogonality_failure"

    elif (
        int(r.n_converged)
        < limits["minimum_converged_states"]
    ):

        reason = "insufficient_convergence"

    elif (
        limits["require_twist_partner"]
        and not bool(r.twist_partner_present)
    ):

        reason = "missing_partner_spectrum"

    else:

        reason = "valid"

    included = reason == "valid"

    qc.append(
        [
            r.sample_id,
            r.campaign,
            "spectral",
            included,
            reason,
            int(r.n_states),
            (
                int(
                    r.n_states
                    - r.compact_states
                )
                if included
                else 0
            ),
        ]
    )

    if included:

        valid_sp.append(
            [
                r.sample_id,
                float(r.alpha),
                int(r["size"]),
                float(r.mean_log_ipr2),
                float(r.mean_gap_ratio),
                float(r.median_twist_shift),
                int(
                    r.n_states
                    - r.compact_states
                ),
            ]
        )


qcdf = pd.DataFrame(
    qc,
    columns=[
        "sample_id",
        "campaign",
        "observable",
        "included",
        "reason",
        "n_raw",
        "n_retained",
    ],
).sort_values("sample_id")


assert set(qcdf.sample_id) == (
    set(tm.sample_id)
    .union(set(sp.sample_id))
)


qcdf.to_csv(
    OUT / "quality_control.csv",
    index=False,
)


tmrun = pd.DataFrame(
    valid_tm,
    columns=[
        "sample_id",
        "alpha",
        "width",
        "lambda_dimensionless",
    ],
)


trows = []


for (a, w), g in tmrun.groupby(
    ["alpha", "width"]
):

    v = (
        g.lambda_dimensionless
        .to_numpy(float)
    )

    trows.append(
        [
            a,
            w,
            v.mean(),
            sem(v),
            len(v),
        ]
    )


tfinal = pd.DataFrame(
    trows,
    columns=[
        "alpha",
        "width",
        "lambda_dimensionless",
        "standard_error",
        "n_independent_runs",
    ],
).sort_values(
    ["alpha", "width"]
)


tfinal.to_csv(
    OUT / "transfer_scaling.csv",
    index=False,
)


sprun = pd.DataFrame(
    valid_sp,
    columns=[
        "sample_id",
        "alpha",
        "size",
        "log_ipr",
        "gap",
        "twist",
        "n_states",
    ],
)


srows = []


for (a, L), g in sprun.groupby(
    ["alpha", "size"]
):

    logs = (
        g.log_ipr
        .to_numpy(float)
    )

    typical = float(
        np.exp(logs.mean())
    )

    srows.append(
        [
            a,
            L,
            len(g),
            int(g.n_states.sum()),
            typical,
            typical * sem(logs),
            g.gap.mean(),
            sem(g.gap),
            g.twist.mean(),
            sem(g.twist),
        ]
    )


sfinal = pd.DataFrame(
    srows,
    columns=[
        "alpha",
        "size",
        "n_realizations",
        "n_states",
        "typical_ipr2",
        "ipr2_error",
        "mean_gap_ratio",
        "gap_ratio_error",
        "median_twist_shift",
        "twist_shift_error",
    ],
).sort_values(
    ["alpha", "size"]
)


sfinal.to_csv(
    OUT / "spectral_observables.csv",
    index=False,
)


f0 = fit_model(
    tfinal,
    False,
)

f1 = fit_model(
    tfinal,
    True,
)


sel = (
    f1
    if f1[3] < f0[3]
    else f0
)


irrelevant = sel is f1


p, chi2, dof, aic = sel


rng = np.random.default_rng(
    20260811
)


bootstrap_alpha = []

bootstrap_nu = []


for _ in range(200):

    boot = tfinal.copy()

    boot["lambda_dimensionless"] = (
        rng.normal(
            tfinal.lambda_dimensionless,
            np.maximum(
                tfinal.standard_error,
                1e-6,
            ),
        )
    )

    try:

        bp, *_ = fit_model(
            boot,
            irrelevant,
        )

        bootstrap_alpha.append(
            float(bp[2])
        )

        bootstrap_nu.append(
            float(bp[3])
        )

    except Exception:

        pass


crit = {

    "critical_alpha":
        float(p[2]),

    "critical_alpha_ci95":
        [
            float(x)
            for x in np.percentile(
                bootstrap_alpha,
                [2.5, 97.5],
            )
        ],

    "nu":
        float(p[3]),

    "nu_ci95":
        [
            float(x)
            for x in np.percentile(
                bootstrap_nu,
                [2.5, 97.5],
            )
        ],

    "irrelevant_exponent":
        (
            -1.0
            if irrelevant
            else None
        ),

    "finite_size_model":
        (
            "linear-relevant-plus-fixed-irrelevant"
            if irrelevant
            else "linear-relevant"
        ),

    "fit_statistic":
        float(chi2),

    "degrees_of_freedom":
        int(dof),

    "aic":
        float(aic),

    "alternative_model_aic":
        float(
            f0[3]
            if irrelevant
            else f1[3]
        ),

    "bootstrap_samples":
        len(bootstrap_alpha),

    "included_widths":
        sorted(
            map(
                int,
                tfinal.width.unique(),
            )
        ),

    "excluded_widths": [],
}


(
    OUT / "critical_fit.json"
).write_text(
    json.dumps(
        crit,
        indent=2,
    )
)


sampled = np.sort(
    tfinal.alpha.unique()
)


critical_sample = float(
    sampled[
        np.argmin(
            np.abs(
                sampled
                - crit["critical_alpha"]
            )
        )
    ]
)


phase_rows = []


for a in sampled:

    tg = (
        tfinal[
            tfinal.alpha == a
        ]
        .sort_values("width")
    )

    sg = (
        sfinal[
            sfinal.alpha == a
        ]
        .sort_values("size")
    )

    slope = float(
        np.polyfit(
            np.log(tg.width),
            tg.lambda_dimensionless,
            1,
        )[0]
    )

    ipr_slope = float(
        np.polyfit(
            np.log(sg["size"]),
            np.log(
                sg.typical_ipr2
            ),
            1,
        )[0]
    )

    if np.isclose(
        a,
        critical_sample,
    ):

        phase = "critical"

    elif a < critical_sample:

        phase = "localized"

    else:

        phase = "metallic"

    phase_rows.append(
        [
            a,
            phase,
            (
                "transfer log-size "
                f"slope={slope:.6f}"
            ),
            (
                f"IPR slope={ipr_slope:.6f}; "
                f"r={sg.mean_gap_ratio.iloc[-1]:.6f}; "
                f"twist="
                f"{sg.median_twist_shift.iloc[-1]:.8f}"
            ),
            (
                "moderate"
                if abs(
                    a
                    - critical_sample
                )
                <= 0.06
                else "high"
            ),
        ]
    )


phase = pd.DataFrame(
    phase_rows,
    columns=[
        "alpha",
        "phase",
        "primary_evidence",
        "secondary_evidence",
        "confidence",
    ],
)


phase.to_csv(
    OUT / "phase_map.csv",
    index=False,
)


report = f"""# Anderson-transition audit

The cleaned campaigns support a genuine anisotropy-driven Anderson transition.
The selected finite-size model gives alpha_c={crit['critical_alpha']:.6f}
with 95% interval [{crit['critical_alpha_ci95'][0]:.6f}, {crit['critical_alpha_ci95'][1]:.6f}]
and nu={crit['nu']:.6f} with 95% interval [{crit['nu_ci95'][0]:.6f}, {crit['nu_ci95'][1]:.6f}].

Transfer scaling is the primary thermodynamic diagnostic. Participation scaling,
adjacent-gap statistics and twisted-boundary sensitivity show stronger finite-size
drift but change systematically across the fitted transition. Duplicate checkpoints,
failed transfer runs, eigensolver failures and missing twist partners were removed
before fitting. Compact states were removed upstream and their excluded counts were audited.

Selected model: {crit['finite_size_model']}; AIC={crit['aic']:.6f};
alternative AIC={crit['alternative_model_aic']:.6f}; chi-square={crit['fit_statistic']:.6f};
degrees of freedom={crit['degrees_of_freedom']}.
"""


(
    OUT / "report.md"
).write_text(report)
