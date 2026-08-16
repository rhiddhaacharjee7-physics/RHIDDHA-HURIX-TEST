from pathlib import Path

import os
import json

import numpy as np
import pandas as pd


DATA = Path(
    os.environ.get(
        "DATA_DIR",
        "/app/data",
    )
)

OUT = Path(
    os.environ.get(
        "OUTPUT_DIR",
        "/app/output",
    )
)


required = [
    "quality_control.csv",
    "transfer_scaling.csv",
    "spectral_observables.csv",
    "critical_fit.json",
    "phase_map.csv",
    "summary.md",
]


for filename in required:

    assert (
        OUT / filename
    ).is_file(), f"missing {filename}"


limits = json.loads(
    (
        DATA
        / "validity_limits.json"
    ).read_text()
)


tm = pd.read_csv(
    DATA / "transfer_runs.csv",
    keep_default_na=False,
)


sp = pd.read_csv(
    DATA / "spectral_runs.csv"
)


q = pd.read_csv(
    OUT / "quality_control.csv"
).set_index("sample_id")


assert len(q) == len(tm) + len(sp)


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

    assert (
        q.loc[
            r.sample_id,
            "reason",
        ]
        == reason
    )


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
        and not bool(
            r.twist_partner_present
        )
    ):

        reason = "missing_partner_spectrum"

    else:

        reason = "valid"

    assert (
        q.loc[
            r.sample_id,
            "reason",
        ]
        == reason
    )


valid = tm[
    q.loc[
        tm.sample_id,
        "included",
    ].to_numpy(bool)
].copy()


valid[
    "lambda_dimensionless"
] = (
    1
    / (
        valid.width
        * valid.mean_gamma
    )
)


expected_transfer = []


for (a, w), g in valid.groupby(
    ["alpha", "width"]
):

    values = (
        g.lambda_dimensionless
        .to_numpy(float)
    )

    error = (
        values.std(ddof=1)
        / np.sqrt(len(values))
    )

    expected_transfer.append(
        [
            a,
            w,
            values.mean(),
            error,
            len(values),
        ]
    )


expected_transfer = pd.DataFrame(
    expected_transfer,
    columns=[
        "alpha",
        "width",
        "lambda_dimensionless",
        "standard_error",
        "n_independent_runs",
    ],
).sort_values(
    ["alpha", "width"]
).reset_index(drop=True)


submitted_transfer = pd.read_csv(
    OUT / "transfer_scaling.csv"
).sort_values(
    ["alpha", "width"]
).reset_index(drop=True)


assert np.allclose(
    expected_transfer[
        [
            "alpha",
            "lambda_dimensionless",
            "standard_error",
        ]
    ],
    submitted_transfer[
        [
            "alpha",
            "lambda_dimensionless",
            "standard_error",
        ]
    ],
    rtol=1e-8,
    atol=1e-10,
)


assert np.array_equal(
    expected_transfer[
        [
            "width",
            "n_independent_runs",
        ]
    ].to_numpy(int),
    submitted_transfer[
        [
            "width",
            "n_independent_runs",
        ]
    ].to_numpy(int),
)


valid_sp = sp[
    q.loc[
        sp.sample_id,
        "included",
    ].to_numpy(bool)
].copy()


expected_spectral = []


for (a, L), g in valid_sp.groupby(
    ["alpha", "size"]
):

    logs = (
        g.mean_log_ipr2
        .to_numpy(float)
    )

    typical = float(
        np.exp(
            logs.mean()
        )
    )

    ipr_error = (
        typical
        * logs.std(ddof=1)
        / np.sqrt(len(logs))
    )

    gaps = (
        g.mean_gap_ratio
        .to_numpy(float)
    )

    twists = (
        g.median_twist_shift
        .to_numpy(float)
    )

    expected_spectral.append(
        [
            a,
            L,
            len(g),
            int(
                (
                    g.n_states
                    - g.compact_states
                ).sum()
            ),
            typical,
            ipr_error,
            gaps.mean(),
            (
                gaps.std(ddof=1)
                / np.sqrt(len(gaps))
            ),
            twists.mean(),
            (
                twists.std(ddof=1)
                / np.sqrt(len(twists))
            ),
        ]
    )


expected_spectral = pd.DataFrame(
    expected_spectral,
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
).reset_index(drop=True)


submitted_spectral = pd.read_csv(
    OUT / "spectral_observables.csv"
).sort_values(
    ["alpha", "size"]
).reset_index(drop=True)


assert np.allclose(
    expected_spectral[
        [
            "alpha",
            "typical_ipr2",
            "ipr2_error",
            "mean_gap_ratio",
            "gap_ratio_error",
            "median_twist_shift",
            "twist_shift_error",
        ]
    ],
    submitted_spectral[
        [
            "alpha",
            "typical_ipr2",
            "ipr2_error",
            "mean_gap_ratio",
            "gap_ratio_error",
            "median_twist_shift",
            "twist_shift_error",
        ]
    ],
    rtol=1e-8,
    atol=1e-12,
)


assert np.array_equal(
    expected_spectral[
        [
            "size",
            "n_realizations",
            "n_states",
        ]
    ].to_numpy(int),
    submitted_spectral[
        [
            "size",
            "n_realizations",
            "n_states",
        ]
    ].to_numpy(int),
)


critical = json.loads(
    (
        OUT
        / "critical_fit.json"
    ).read_text()
)


assert (
    0.66
    < critical["critical_alpha"]
    < 0.78
)


assert (
    0.8
    < critical["nu"]
    < 2.5
)


assert (
    critical[
        "critical_alpha_ci95"
    ][0]
    <= critical[
        "critical_alpha"
    ]
    <= critical[
        "critical_alpha_ci95"
    ][1]
)


assert (
    critical["nu_ci95"][0]
    <= critical["nu"]
    <= critical["nu_ci95"][1]
)


assert (
    critical[
        "bootstrap_samples"
    ]
    >= 150
)


phase = pd.read_csv(
    OUT / "phase_map.csv"
).sort_values("alpha")


assert (
    phase.phase
    == "critical"
).sum() == 1


critical_alpha = phase.loc[
    phase.phase == "critical",
    "alpha",
].iloc[0]


assert (
    phase.loc[
        phase.alpha
        < critical_alpha,
        "phase",
    ]
    == "localized"
).all()


assert (
    phase.loc[
        phase.alpha
        > critical_alpha,
        "phase",
    ]
    == "metallic"
).all()


summary = (
    OUT / "summary.md"
).read_text().lower()


for term in [
    "transition",
    "transfer",
    "participation",
    "adjacent-gap",
    "twisted-boundary",
]:

    assert (
        term in summary
    ), f"summary missing {term}"


print("PASS")
