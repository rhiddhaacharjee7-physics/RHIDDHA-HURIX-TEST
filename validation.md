# Validation

Official Harbor validation must be pasted here after running the task in the required local Docker/Harbor environment.

Required commands:

```bash
harbor run -p tasks/physics/condensed-matter/anisotropic-anderson-audit -a oracle
harbor run -p tasks/physics/condensed-matter/anisotropic-anderson-audit -a nop
```

Required results:

- Oracle mean: `1.000`
- NOP mean: `0.000`

Do not replace this section with invented output. Paste the complete terminal output from the actual runs.
