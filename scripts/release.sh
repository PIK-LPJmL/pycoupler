#!/bin/bash
# Script to create a local release (update CITATION.cff, commit, tag) without publishing to PyPI

VERSION=$1
if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 1.5.25"
    exit 1
fi

echo "Creating local release for version: $VERSION"

# 1. Update CITATION.cff with the specified version
echo "Updating CITATION.cff..."
python3 scripts/update_citation.py "$VERSION"

# Check if CITATION.cff was updated
if ! git diff --quiet CITATION.cff; then
    echo "CITATION.cff updated, committing changes..."
    git add CITATION.cff
    git commit -m "Update CITATION.cff to version $VERSION"
    echo "CITATION.cff changes committed successfully."
else
    echo "No changes to CITATION.cff needed."
fi

# 2. Create the tag
echo "Creating tag v$VERSION..."
git tag "v$VERSION"

# 3. Show what was done
echo ""
echo "Local release completed for version $VERSION!"
echo ""
echo "To push to repository, run:"
echo "  git push origin main --tags"
echo ""
echo "The CI pipeline will automatically:"
echo "  - Build the package"
echo "  - Upload to PyPI"
echo "  - Create GitHub release"
