# Security Policy

## Supported versions

Vega is pre-1.0 and under active development. Security fixes are applied to the
`main` branch only. There is no long-term support branch yet.

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use one of the following:

- GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
  (Security tab → "Report a vulnerability"), or
- Open a minimal issue asking maintainers for a private contact channel, without
  disclosing details.

Please include:

- A description of the issue and its impact
- Steps to reproduce (proof-of-concept if possible)
- Affected component (`vega_server`, `vega_worker`, PyVega client, etc.)
- Your environment (OS, compiler, Python version)

## Scope and known limitations

Vega is research-grade software. Be aware that, by design in the current
release:

- `vega_server` and `vega_worker` provide **no authentication or transport
  encryption**. Do not expose them on untrusted networks.
- The server executes **arbitrary user-provided Python** (UDFs are unpickled and
  run). Only connect trusted clients.
- Cluster peers trust gossip membership without verification.

These are not separately tracked as vulnerabilities; treat Vega as you would any
service that runs untrusted code, and isolate it accordingly.

## Response

We aim to acknowledge reports within a reasonable time and will coordinate a
fix and disclosure timeline with the reporter.
