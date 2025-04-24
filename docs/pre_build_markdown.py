#!/usr/bin/env python
"""
Script to build markdown documentation for ReadTheDocs.
This script is automatically run by ReadTheDocs before the main documentation build.
"""

import os
import sys
import subprocess
import shutil
import re


def update_conf_py():
    """Update conf.py to include the markdown file in html_extra_path."""
    print("Updating conf.py to include the markdown file in html_extra_path...")
    
    conf_path = "docs/conf.py"
    with open(conf_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if html_extra_path is already set
    if "html_extra_path = " in content:
        # Replace the existing html_extra_path
        pattern = r"html_extra_path = \[([^\]]*)\]"
        if re.search(pattern, content):
            # Add to existing list
            if "'../cobaya_docs_combined.md'" not in content and '"../cobaya_docs_combined.md"' not in content:
                content = re.sub(
                    pattern,
                    lambda m: m.group(0)[:-1] + (", '../cobaya_docs_combined.md'" if m.group(1) else "'../cobaya_docs_combined.md'") + "]",
                    content
                )
        else:
            # Replace the commented line
            content = content.replace(
                "# html_extra_path = []",
                "html_extra_path = ['../cobaya_docs_combined.md']"
            )
    else:
        # Add html_extra_path after html_static_path
        content = content.replace(
            "html_static_path = ['theme_customisation']",
            "html_static_path = ['theme_customisation']\n\n# Add any extra paths that contain custom files\nhtml_extra_path = ['../cobaya_docs_combined.md']"
        )
    
    # Write the updated content back to the file
    with open(conf_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print("conf.py updated successfully.")


def main():
    """Build markdown documentation before the Sphinx build."""
    print("Building markdown documentation for ReadTheDocs...")
    
    # Create _build directory
    os.makedirs("_build", exist_ok=True)
    
    # Run the build_docs_to_markdown.py script
    build_script = "docs/build_docs_to_markdown.py"
    output_file = "cobaya_docs_combined.md"
    
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
        
        # Update conf.py to include the markdown file
        update_conf_py()
    else:
        print(f"ERROR: Failed to generate markdown file at {output_file}")
    
    print("Pre-build markdown documentation process completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
