# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Python Version Support

| Python Version | Supported          |
| -------------- | ------------------ |
| 3.12           | :white_check_mark: |
| 3.11           | :white_check_mark: |
| 3.10           | :white_check_mark: |
| 3.9            | :white_check_mark: |
| < 3.9          | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability within analogtivation, please send an email to riosanjuan314@gmail.com. All security vulnerabilities will be promptly addressed.

Please include the following information:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

## Security Update Process

1. Security patches will be released as soon as possible after verification
2. All security updates will be documented in the CHANGELOG
3. Critical vulnerabilities will trigger immediate patch releases
4. We use Dependabot to monitor and update dependencies automatically

## Best Practices

When using analogtivation:
- Always use the latest version
- Keep your Python environment updated
- Monitor your dependencies using tools like `pip-audit`
- Never use activation functions with untrusted input data
- Be aware that time-based activations may behave differently across time zones
