#!/usr/bin/env python
"""
Script to build markdown documentation for ReadTheDocs.
This script is run by ReadTheDocs before the main documentation build.
"""

import os
import sys
import subprocess
import traceback


def update_conf_py():
    """Update conf.py to include the markdown file in html_static_path."""
    print("Updating conf.py to include the markdown file in html_static_path...")

    conf_path = "docs/conf.py"
    with open(conf_path, "r", encoding="utf-8") as f:
        content = f.read()

    # The path to the markdown file relative to the conf.py directory
    md_path = "cobaya_docs_combined.md"

    # Check if html_static_path is already set
    if "html_static_path = " in content:
        # Replace the existing html_static_path
        if "html_static_path = []" in content:
            # Replace empty list
            content = content.replace(
                "html_static_path = []",
                f"html_static_path = ['{md_path}']"
            )
        elif f"'{md_path}'" not in content and f'"{md_path}"' not in content:
            # Add to existing list if not already there
            content = content.replace(
                "html_static_path = [",
                f"html_static_path = ['{md_path}', "
            )
    else:
        # Add html_static_path if not present
        content += f"\n\n# Extra files to include\nhtml_static_path = ['{md_path}']\n"

    # Write the updated content back to the file
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("conf.py updated successfully.")


def main():
    """Build markdown documentation before the Sphinx build."""
    print("Building markdown documentation for ReadTheDocs...")

    try:
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
            check=True,  # This will raise an exception if the command fails
            capture_output=True,
            text=True
        )

        # Print output for debugging
        print(f"Build script stdout: {result.stdout}")
        if result.stderr:
            print(f"Build script stderr: {result.stderr}")

        # Check if the markdown file was generated
        if not os.path.exists(output_file):
            print(f"ERROR: Failed to generate markdown file at {output_file}")
            return 1

        print(f"Markdown file generated at: {output_file}")

        # Verify the file exists and has content
        file_size = os.path.getsize(output_file)
        print(f"Markdown file size: {file_size} bytes")
        if file_size == 0:
            print("ERROR: Markdown file is empty")
            return 1

        # Update conf.py to include the markdown file
        update_conf_py()

        print("Pre-build markdown documentation completed successfully.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Build script failed with return code {e.returncode}")
        print(f"Output: {e.output}")
        print(f"Error: {e.stderr}")
        return 1
    except Exception as e:
        print(f"ERROR: An exception occurred during the build process: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
