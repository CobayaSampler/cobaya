#!/usr/bin/env python
"""
Script to build markdown documentation for ReadTheDocs.
This script is run by ReadTheDocs before the main documentation build.
"""

import os
import sys
import subprocess
import re


def update_conf_py():
    """Update conf.py to include the markdown file in html_extra_path."""
    print("Updating conf.py to include the markdown file...")

    conf_path = "docs/conf.py"
    with open(conf_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The path to the markdown file relative to the conf.py directory
    md_path = "cobaya_docs_combined.md"

    # Check if html_extra_path is already set
    if "html_extra_path = " in content:
        # Replace the existing html_extra_path
        pattern = r"html_extra_path = \[([^\]]*)\]"
        if re.search(pattern, content):
            # Add to existing list if not already there
            if f"'{md_path}'" not in content and f'"{md_path}"' not in content:
                content = re.sub(
                    pattern,
                    lambda m: m.group(0)[:-1] +
                             (f", '{md_path}'" if m.group(1) else f"'{md_path}'") +
                             "]",
                    content
                )
        else:
            # Replace the commented line
            content = content.replace(
                "# html_extra_path = []",
                f"html_extra_path = ['{md_path}']"
            )
    else:
        # Add html_extra_path after html_static_path
        content = content.replace(
            "html_static_path = ['theme_customisation']",
            "html_static_path = ['theme_customisation']\n\n" +
            "# Add any extra paths that contain custom files\n" +
            f"html_extra_path = ['{md_path}']"
        )

    # Write the updated content back to the file
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("conf.py updated successfully.")


def main():
    """Build markdown documentation before the Sphinx build."""
    print("Building markdown documentation for ReadTheDocs...")

    # Use docs directory for output
    output_dir = "docs"
    output_file = os.path.join(output_dir, "cobaya_docs_combined.md")

    # Run the build_docs_to_markdown.py script
    build_script = "docs/build_docs_to_markdown.py"

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

    # Check if the build was successful
    if result.returncode != 0:
        print(f"ERROR: Build script failed with return code {result.returncode}")
        return 1

    # Check if the markdown file was generated
    if not os.path.exists(output_file):
        print(f"ERROR: Failed to generate markdown file at {output_file}")
        return 1

    print(f"Markdown file generated at: {output_file}")

    # Update conf.py to include the markdown file
    update_conf_py()

    print("Pre-build markdown documentation process completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
