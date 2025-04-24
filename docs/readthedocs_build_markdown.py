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
    # ReadTheDocs sets READTHEDOCS_OUTPUT/html for the HTML output
    output_dir = os.environ.get('READTHEDOCS_OUTPUT', '')
    if output_dir:
        output_dir = os.path.join(output_dir, 'html')
    else:
        # Fallback to a local directory if not running on ReadTheDocs
        output_dir = '_build/html'

    print(f"Output directory: {output_dir}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create _build directory
    os.makedirs("_build", exist_ok=True)

    # Run the build_docs_to_markdown.py script
    build_script = "docs/build_docs_to_markdown.py"
    output_file = "_build/cobaya_docs_combined.md"

    print(f"Running {build_script} to generate {output_file}")

    # Run the script
    result = subprocess.run(
        [
            sys.executable,
            build_script,
            "--exclude", "cluster_amazon,devel",
            "--output", output_file
        ],
        check=False,
        capture_output=True,
        text=True
    )

    # Print output for debugging
    print(f"Build script stdout: {result.stdout}")
    if result.stderr:
        print(f"Build script stderr: {result.stderr}")

    # Check if the markdown file was generated
    if os.path.exists(output_file):
        print(f"Markdown file generated at: {output_file}")

        # Copy the file directly to the output directory root
        # This makes it accessible at the URL: /cobaya_docs_combined.md
        dest_file = os.path.join(output_dir, "cobaya_docs_combined.md")
        shutil.copy2(output_file, dest_file)
        print(f"Copied markdown file to: {dest_file}")
    else:
        print(f"ERROR: Failed to generate markdown file at {output_file}")

    print("Markdown documentation process completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
