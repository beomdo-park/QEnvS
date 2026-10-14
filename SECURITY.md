# Security policy

## Supported versions

Security fixes are released for the latest minor version on PyPI. QEnvS is in the 0.x
series; upgrade to the newest release before reporting.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
[Security advisories, "Report a vulnerability"](https://github.com/beomdo-park/QEnvS/security/advisories/new).
Do not open a public issue. You should get a first answer within 7 days.

## Scope notes

* QEnvS performs numerical simulation only. It opens no network connections and reads or
  writes files only where you tell it to (for example `examples/benchmark_battery.py --out`).
* `qenvs validate --model package.module:attr` imports and executes the named Python module,
  in the same way `pytest` or `python -m` do. Only point it at code you trust.
* Releases are built in GitHub Actions and published to PyPI through Trusted Publishing; each
  release publishes PEP 740 attestations for the wheel and the sdist to PyPI, and the same
  attestation files (`*.publish.attestation`) are attached to the GitHub release.
