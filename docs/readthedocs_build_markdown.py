#!/usr/bin/env python
"""
Script to build markdown documentation for ReadTheDocs.
This script is automatically run by ReadTheDocs after the main documentation build.
"""

import os
import sys
import subprocess
import shutil


def main():
    """Build markdown documentation and copy it to the HTML output directory."""
    print("Building markdown documentation for ReadTheDocs...")

    # Get the output directory from environment variables
    output_dir = os.environ.get('READTHEDOCS_OUTPUT', 'docs/_readthedocs/html')

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Run the build_docs_to_markdown.py script
    subprocess.run(
        [
            sys.executable,
            "docs/build_docs_to_markdown.py",
            "--exclude", "cluster_amazon,devel",
            "--output", "docs/_build/cobaya_docs_combined.md"
        ],
        check=True
    )

    # Copy the markdown file to the output directory
    shutil.copy2(
        "docs/_build/cobaya_docs_combined.md",
        os.path.join(output_dir, "cobaya_docs_combined.md")
    )

    # Also copy to the static directory for the HTML link to work
    static_dir = os.path.join(output_dir, "_static")
    os.makedirs(static_dir, exist_ok=True)
    shutil.copy2(
        "docs/_build/cobaya_docs_combined.md",
        os.path.join(static_dir, "cobaya_docs_combined.md")
    )

    print("Markdown documentation built and copied to the output directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
