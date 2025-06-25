#!/usr/bin/env python
"""Run the test suite with various options."""

import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run analogtivation tests")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Run only core implementation tests"
    )
    parser.add_argument(
        "--framework",
        choices=["tensorflow", "torch", "jax"],
        help="Run tests for a specific framework"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Build pytest command
    cmd = ["pytest"]
    
    if args.coverage:
        cmd.extend(["--cov=analogtivation", "--cov-report=html", "--cov-report=term-missing"])
    
    if args.verbose:
        cmd.append("-vv")
    
    if args.core_only:
        cmd.append("tests/test_core_*.py")
    elif args.framework:
        cmd.append(f"tests/test_{args.framework}_*.py")
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=".")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()