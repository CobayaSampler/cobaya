#!/usr/bin/env python
"""
Script to build markdown documentation for ReadTheDocs.
This script is run by ReadTheDocs before the main documentation build.
"""

import os
import sys
import subprocess
import re
import shutil
import traceback


def create_html_version(md_file):
    """Create an HTML version of the markdown file."""
    print("Creating HTML version of the markdown file...")

    html_file = md_file.replace('.md', '.html')

    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Create a simple HTML wrapper for the markdown content
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Cobaya Documentation for LLMs</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
        h1, h2, h3, h4, h5, h6 {{ margin-top: 24px; margin-bottom: 16px; }}
        a {{ color: #0366d6; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Cobaya Documentation for LLMs</h1>
    <p>This is the complete Cobaya documentation in a format suitable for use with Large Language Models.</p>
    <hr>
    <pre>{md_content}</pre>
</body>
</html>
"""

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML version created at: {html_file}")
        return html_file
    except Exception as e:
        print(f"ERROR creating HTML version: {e}")
        return None


def update_conf_py():
    """Update conf.py to include files in html_static_path."""
    print("Updating conf.py to include files in html_static_path...")

    conf_path = "docs/conf.py"
    with open(conf_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Create a _static directory if it doesn't exist
    static_dir = os.path.join("docs", "_static")
    os.makedirs(static_dir, exist_ok=True)

    # Check if html_static_path is already set
    if "html_static_path = " in content:
        # Make sure theme_customisation is included
        if "html_static_path = ['theme_customisation']" in content:
            # Already set correctly
            pass
        elif "html_static_path = []" in content:
            # Replace empty list
            content = content.replace(
                "html_static_path = []",
                "html_static_path = ['theme_customisation']"
            )
        else:
            # Add theme_customisation if not already there
            if "'theme_customisation'" not in content and '"theme_customisation"' not in content:
                pattern = r"html_static_path = \[([^\]]*)\]"
                content = re.sub(
                    pattern,
                    lambda m: m.group(0)[:-1] +
                             (", 'theme_customisation'" if m.group(1) else "'theme_customisation'") +
                             "]",
                    content
                )
    else:
        # Add html_static_path if not present
        content += "\n\n# Path to static files\nhtml_static_path = ['theme_customisation']\n"

    # Write the updated content back to the file
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Verify the update by reading the file again
    with open(conf_path, "r", encoding="utf-8") as f:
        updated_content = f.read()

    # Check if html_static_path is in the updated content
    if "html_static_path = ['theme_customisation']" in updated_content or \
       'html_static_path = ["theme_customisation"]' in updated_content or \
       "'theme_customisation'" in updated_content and "html_static_path = [" in updated_content:
        print("conf.py updated successfully with html_static_path.")
    else:
        print("ERROR: Failed to update conf.py with html_static_path.")
        print("Current content of conf.py:")
        print("---")
        print(updated_content)
        print("---")
        raise RuntimeError("Failed to update conf.py with html_static_path.")


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
            print(f"ERROR: Build script failed with return code "
                  f"{result.returncode}")
            return 1

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

        # Create HTML version of the markdown file
        html_file = create_html_version(output_file)
        if not html_file:
            print("WARNING: Failed to create HTML version of the markdown file")

        # Update conf.py to ensure html_static_path is set correctly
        update_conf_py()

        # Copy files to _static directory
        static_dir = os.path.join("docs", "_static")
        os.makedirs(static_dir, exist_ok=True)

        # Copy markdown file to _static
        static_md_file = os.path.join(static_dir, "cobaya_docs_combined.md")
        shutil.copy2(output_file, static_md_file)
        print(f"Copied markdown file to: {static_md_file}")

        # Copy HTML file to _static if it exists
        if html_file and os.path.exists(html_file):
            static_html_file = os.path.join(
                static_dir, "cobaya_docs_combined.html")
            shutil.copy2(html_file, static_html_file)
            print(f"Copied HTML file to: {static_html_file}")

        print("Pre-build markdown documentation completed successfully.")
        return 0
    except Exception as e:
        print(f"ERROR: An exception occurred during the build process: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
