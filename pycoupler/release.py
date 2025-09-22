#!/usr/bin/env python3
"""
Release utility for pycoupler, pycopancore, and pycopanlpjml packages.

This module provides a unified release script that can be used across all three packages
to handle version updates, code quality checks, and Git operations.

Usage:
    python3 -m pycoupler.release <version>
    python3 -m pycoupler.release 1.5.25

The script will:
1. Format code with black
2. Run tests with pytest
3. Run linting with flake8
4. Update CITATION.cff (if needed)
5. Commit changes
6. Create Git tag
"""

import sys
import subprocess
from pathlib import Path


def get_package_name():
    """Detect the current package name from the directory structure."""
    current_dir = Path.cwd()

    # Check if we're in a package directory
    if (current_dir / "pyproject.toml").exists():
        try:
            with open("pyproject.toml", "r") as f:
                content = f.read()
                # Extract project name from pyproject.toml
                for line in content.split("\n"):
                    if line.strip().startswith("name = "):
                        return line.split('"')[1]
        except Exception:
            pass

    # Fallback: use directory name
    return current_dir.name


def get_current_branch():
    """Get the current Git branch name."""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "main"  # fallback


def run_command(cmd, description, fail_on_error=True):
    """Run a command and handle errors."""
    print(f"{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"{description} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        if fail_on_error:
            print(f"Error: {description} failed!")
            sys.exit(1)
        else:
            print(f"Warning: {description} failed: {e}")
            return False


def update_citation_file(version, date=None):
    """Update CITATION.cff file with the given version."""
    citation_file = Path("CITATION.cff")
    if not citation_file.exists():
        print("No CITATION.cff file found, skipping...")
        return False

    try:
        with open(citation_file, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        updated = False

        for i, line in enumerate(lines):
            if line.startswith("version:"):
                old_version = line.split(":", 1)[1].strip()
                if old_version != version:
                    lines[i] = f"version: {version}"
                    updated = True
                    print(f"Updated version: {old_version} -> {version}")
            elif line.startswith("date-released:") and date:
                old_date = line.split(":", 1)[1].strip()
                if old_date != date:
                    lines[i] = f"date-released: {date}"
                    updated = True
                    print(f"Updated date: {old_date} -> {date}")

        if updated:
            with open(citation_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            return True
        else:
            return False

    except Exception as e:
        print(f"Error updating CITATION.cff: {e}")
        return False


def main():
    """Main release function."""
    if len(sys.argv) < 2 or sys.argv[1] in ["--help", "-h"]:
        print("Usage: python3 -m pycoupler.release <version>")
        print("Example: python3 -m pycoupler.release 1.5.25")
        print("")
        print("This script will:")
        print("  - Format code with black")
        print("  - Run tests with pytest")
        print("  - Run linting with flake8")
        print("  - Update CITATION.cff (if needed)")
        print("  - Commit changes")
        print("  - Create Git tag")
        print("")
        print("Prerequisites: Install dev dependencies first:")
        print("  pip install -e .[dev]")
        sys.exit(0)

    version = sys.argv[1]
    package_name = get_package_name()
    current_branch = get_current_branch()

    print(f"Creating local release for {package_name} version: {version}")
    print(f"Current branch: {current_branch}")

    # 1. Format code with black
    run_command("python3 -m black ./", "Formatting code with black")

    # 2. Run tests with pytest
    run_command("python3 -m pytest", "Running tests with pytest")

    # 3. Run linting with flake8
    run_command("python3 -m flake8", "Running linting with flake8")

    # 4. Update CITATION.cff and commit changes (only after all checks pass)
    print("Updating CITATION.cff...")
    updated = update_citation_file(version)

    if updated:
        print("CITATION.cff has been updated!")
    else:
        print("No updates needed for CITATION.cff")

    # Check if there are changes to commit
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "CITATION.cff"], capture_output=True, text=True
        )
        has_changes = result.returncode != 0
    except subprocess.CalledProcessError:
        has_changes = False

    if has_changes:
        print("CITATION.cff updated, committing changes...")
        run_command("git add CITATION.cff", "Adding CITATION.cff to staging")
        run_command(
            f'git commit -m "Release version {version}\\n\\n- Update CITATION.cff to version {version}"',  # noqa: E501
            "Committing CITATION.cff changes",
        )
        print("CITATION.cff changes committed successfully.")
    else:
        print("No changes to CITATION.cff needed.")

    # 5. Create the tag
    print(f"Creating tag v{version}...")
    run_command(f'git tag "v{version}"', "Creating Git tag")

    # 6. Show what was done
    print("")
    print(f"Local release completed for {package_name} version {version}!")
    print("")
    print("To push to repository, run:")
    print(f"  git push origin {current_branch} --tags")
    print("")
    print("The CI pipeline will automatically:")
    print("  - Run tests with pytest and coverage")
    print("  - Check code formatting with black")
    print("  - Lint code with flake8")
    print("  - Build the package")
    print("  - Validate package with twine check")
    print("  - Test package installation")
    print("  - Upload to PyPI (if tag push)")
    print("  - Create GitHub release (if tag push)")
    print("  - Create Zenodo release (if tag push)")


if __name__ == "__main__":
    main()
