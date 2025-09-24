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
import re
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


def get_current_version():
    """Get the current version from the latest Git tag."""
    try:
        # Get the latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True,
        )
        tag = result.stdout.strip()
        # Remove 'v' prefix if present
        version = tag.lstrip("v")
        return version
    except subprocess.CalledProcessError:
        # No tags exist, return None
        return None


def parse_version(version_string):
    """Parse a version string into major, minor, patch components."""
    # Match semantic versioning pattern (major.minor.patch)
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_string)
    if not match:
        return None
    return tuple(int(x) for x in match.groups())


def is_valid_version_increment(current_version, new_version):
    """Check if the new version is a valid increment from the current version."""
    if current_version is None:
        # No current version, any valid semantic version is allowed
        return parse_version(new_version) is not None

    current_parts = parse_version(current_version)
    new_parts = parse_version(new_version)

    if current_parts is None or new_parts is None:
        return False

    # Check if it's the same version (allowed for re-releasing)
    if current_parts == new_parts:
        return True

    # Check if it's a valid increment
    # Valid increments: patch (0.0.1), minor (0.1.0), or major (1.0.0)
    if new_parts[0] > current_parts[0]:
        # Major version increment - must be exactly +1
        return (
            new_parts[0] == current_parts[0] + 1
            and new_parts[1] == 0
            and new_parts[2] == 0
        )
    elif new_parts[0] == current_parts[0]:
        if new_parts[1] > current_parts[1]:
            # Minor version increment - must be exactly +1
            return new_parts[1] == current_parts[1] + 1 and new_parts[2] == 0
        elif new_parts[1] == current_parts[1]:
            # Patch version increment - must be exactly +1
            return new_parts[2] == current_parts[2] + 1
        else:
            # Minor version went backwards
            return False
    else:
        # Major version went backwards
        return False


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


def delete_tag_if_exists(tag_name):
    """Delete a tag locally and remotely if it exists."""
    print(f"Checking if tag {tag_name} already exists...")

    # Check if tag exists locally (fast)
    try:
        result = subprocess.run(
            ["git", "tag", "-l", tag_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if tag_name in result.stdout:
            print(f"Tag {tag_name} exists locally, deleting...")
            run_command(f"git tag -d {tag_name}", f"Deleting local tag {tag_name}")
        else:
            print(f"Tag {tag_name} does not exist locally")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(f"Tag {tag_name} does not exist locally")

    # Always try to delete remote tag (in case it exists)
    print(f"Deleting remote tag {tag_name} (if it exists)...")
    try:
        run_command(
            f"git push origin :refs/tags/{tag_name}",
            f"Deleting remote tag {tag_name}",
            fail_on_error=False,  # Don't fail if tag doesn't exist remotely
        )
    except Exception:
        # Ignore errors - tag might not exist remotely
        pass

    print(f"Tag {tag_name} cleanup completed.")
    print(
        "Note: You can now push your new commit and tag manually with: "
        "git push origin <branch> --tags"
    )


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
        print("  - Delete existing local tag (if it exists)")
        print("  - Create Git tag")
        print("")
        print("Prerequisites: Install dev dependencies first:")
        print("  pip install -e .[dev]")
        sys.exit(0)

    version = sys.argv[1]
    package_name = get_package_name()
    current_branch = get_current_branch()

    # Validate version format and increment
    current_version = get_current_version()
    if not is_valid_version_increment(current_version, version):
        print(f"Error: Invalid version increment!")
        if current_version:
            print(f"Current version: {current_version}")
            print(f"Requested version: {version}")
            print("")
            print("Valid version increments:")
            current_parts = parse_version(current_version)
            if current_parts:
                major, minor, patch = current_parts
                print(f"  - Same version (re-release): {version}")
                print(f"  - Patch increment: {major}.{minor}.{patch + 1}")
                print(f"  - Minor increment: {major}.{minor + 1}.0")
                print(f"  - Major increment: {major + 1}.0.0")
        else:
            print(
                "No current version found. Any valid semantic version (X.Y.Z) is allowed."  # noqa: E501
            )
        print("")
        print("Examples of valid semantic versioning:")
        print("  - 1.0.0 -> 1.0.1 (patch)")
        print("  - 1.0.0 -> 1.1.0 (minor)")
        print("  - 1.0.0 -> 2.0.0 (major)")
        print("  - 1.0.0 -> 1.0.0 (re-release)")
        sys.exit(1)

    # Check if this is a re-release and ask for confirmation
    if current_version and parse_version(current_version) == parse_version(
        version
    ):  # noqa: E501
        print(
            f"⚠️  WARNING: You are attempting to re-release version {version}"
        )  # noqa: E501
        print(
            f"This will delete the existing tag and create a new one with recent changes."  # noqa: E501
        )
        print("")
        while True:
            response = (
                input(
                    "Are you certain you want to re-release this version? [y/N]: "
                )  # noqa: E501
                .strip()
                .lower()
            )
            if response in ["y", "yes"]:
                print("✅ Re-release confirmed. Proceeding...")
                break
            elif response in ["n", "no", ""]:
                print("❌ Re-release cancelled.")
                sys.exit(0)
            else:
                print("Please answer 'y' for yes or 'n' for no.")

    print(f"Creating local release for {package_name} version: {version}")
    if current_version:
        print(f"Current version: {current_version}")
    else:
        print("No previous version found (first release)")
    print(f"Current branch: {current_branch}")

    # 1. Format code with black
    run_command("python3 -m black ./", "Formatting code with black")

    # 2. Run tests with pytest (if tests exist)
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            check=True,
        )
        if "collected 0 items" in result.stdout:
            print("No tests found, skipping pytest...")
        else:
            run_command("python3 -m pytest", "Running tests with pytest")
    except subprocess.CalledProcessError:
        print("No tests found, skipping pytest...")

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
            ["git", "diff", "--quiet", "CITATION.cff"],
            capture_output=True,
            text=True,  # noqa: E501
        )
        has_changes = result.returncode != 0
    except subprocess.CalledProcessError:
        has_changes = False

    if has_changes:
        print("CITATION.cff updated, committing changes...")
        run_command("git add CITATION.cff", "Adding CITATION.cff to staging")
        run_command(
            f'git commit -m "Version {version}"',
            "Committing CITATION.cff changes",
        )
        print("CITATION.cff changes committed successfully.")
    else:
        print("No changes to CITATION.cff needed.")

    # 5. Delete existing tag if it exists (both locally and remotely)
    tag_name = f"v{version}"
    delete_tag_if_exists(tag_name)

    # 6. Create the tag
    print(f"Creating tag {tag_name}...")
    run_command(f'git tag "{tag_name}"', "Creating Git tag")

    # 7. Show what was done
    print("")
    print(f"Local release completed for {package_name} version {version}!")
    print("")
    print("To push to repository, run:")
    print(f"  git push origin {current_branch} --tags")
    print("")
    print("Note: Remote tag has been automatically deleted if it existed.")
    print("You can now safely push your new commit and tag together.")
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
