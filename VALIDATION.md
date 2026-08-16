
# Validation

This task was validated locally using Harbor with Docker.

## Oracle Validation

Command:

```bash
harbor run -p "." -a oracle
```

Terminal result:

```text
Trials: 1
Exceptions: 0
Mean: 1.000

Reward: 1.0
Count: 1

Total runtime: 21s
```

Status: **PASS**

The Oracle agent successfully completed the task and achieved the required mean reward of `1.000`.

---

## NOP Validation

Command:

```bash
harbor run -p "." -a nop
```

Terminal result:

```text
Trials: 1
Exceptions: 0
Mean: 0.000

Reward: 0.0
Count: 1

Total runtime: 13s
```

Status: **PASS**

The NOP agent produced no valid solution and correctly received the required mean reward of `0.000`.

---

## Final Validation Summary

- Oracle trials: `1`
- Oracle exceptions: `0`
- Oracle mean: `1.000`
- Oracle reward: `1.0`

- NOP trials: `1`
- NOP exceptions: `0`
- NOP mean: `0.000`
- NOP reward: `0.0`

## Overall Status

**PASS**

Oracle Mean = 1.000
NOP Mean    = 0.000
```

The reference solution is accepted by the verifier, while an empty/no-operation submission is rejected.
```
