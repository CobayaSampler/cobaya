#!/usr/bin/env python
"""
Markdown Builder for Cobaya Documentation

This script builds Sphinx documentation in Markdown format and combines it into a single file
for use as context with Large Language Models (LLMs).

It can be used:
1. As a pre-build step in ReadTheDocs
2. Locally to generate markdown documentation
3. In CI/CD pipelines

Usage:
    python markdown_builder.py [--exclude file1,file2,...] [--output output_file] [--no-install]

Options:
    --exclude: Comma-separated list of files to exclude (without .md extension)
    --output: Output file path (default: docs/cobaya_docs_combined.md)
    --no-install: Skip installation of dependencies
"""

import os
import sys
import subprocess
import argparse
import glob
import traceback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Sphinx documentation in Markdown format for LLM context."
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="cluster_amazon,devel",
        help="Comma-separated list of files to exclude (without .md extension)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/cobaya_docs_combined.md",
        help="Output file path",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip installation of dependencies",
    )
    return parser.parse_args()


def install_dependencies():
    """Install required dependencies."""
    print("Installing sphinx-markdown-builder...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "sphinx-markdown-builder"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("Installation successful.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing sphinx-markdown-builder: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        print("\nPlease install it manually with: pip install sphinx-markdown-builder")
        return False


def update_conf_py():
    """Update conf.py to include sphinx_markdown_builder extension and static path."""
    print("Updating conf.py configuration...")
    conf_path = "docs/conf.py"

    with open(conf_path, encoding="utf-8") as f:
        content = f.read()

    updated = False

    # 1. Add sphinx_markdown_builder extension if not already present
    if "sphinx_markdown_builder" not in content:
        print("Adding sphinx_markdown_builder to conf.py...")

        # Check if we already have the try-except block
        if "# Add sphinx_markdown_builder if available" not in content:
            # Find the extensions list
            if "extensions = [" in content and "]" in content:
                extensions_start = content.find("extensions = [")
                extensions_end = content.find("]", extensions_start)

                if extensions_end > extensions_start:
                    # Insert the try-except block after the extensions list
                    new_content = (
                        content[: extensions_end + 1]
                        + "\n\n# Add sphinx_markdown_builder if available (used for LLM context generation)"
                        + "\ntry:"
                        + "\n    import sphinx_markdown_builder"
                        + "\n    extensions.append('sphinx_markdown_builder')"
                        + "\nexcept ImportError:"
                        + "\n    pass"
                        + content[extensions_end + 1 :]
                    )
                    content = new_content
                    updated = True
                else:
                    print("Warning: Could not find the end of extensions list in conf.py")
            else:
                print("Warning: Could not find extensions list in conf.py")
        else:
            print("sphinx_markdown_builder extension block already exists in conf.py")
    else:
        print("sphinx_markdown_builder is already in conf.py")

    # Write the updated content back to the file if changes were made
    if updated:
        with open(conf_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("conf.py updated successfully.")
    else:
        print("No changes needed in conf.py")

    return True


def build_markdown_docs():
    """Build the documentation in Markdown format."""
    print("Building documentation in Markdown format...")
    build_dir = "docs/_build/markdown"

    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)

    # Run sphinx-build
    result = subprocess.run(
        [
            "sphinx-build",
            "-b",
            "markdown",
            "docs",
            build_dir,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Warning: sphinx-build returned non-zero exit code: {result.returncode}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        # Continue anyway as we might still have generated some markdown files

    return build_dir


def combine_markdown_files(build_dir, exclude_files, output_file):
    """Combine Markdown files into a single file with improved structure."""
    print(f"Combining Markdown files into {output_file}...")

    # Get all markdown files
    md_files = sorted(glob.glob(os.path.join(build_dir, "*.md")))

    if not md_files:
        print(f"Error: No markdown files found in {build_dir}")
        return False

    # Convert exclude_files to a set for faster lookup
    exclude_set = {f"{name.strip()}.md" for name in exclude_files if name.strip()}

    # Print excluded files for debugging
    if exclude_set:
        print(f"Excluding the following files: {', '.join(exclude_set)}")

    # Filter out excluded files
    filtered_files = [f for f in md_files if os.path.basename(f) not in exclude_set]

    # Check if any files were actually excluded
    excluded_count = len(md_files) - len(filtered_files)
    if excluded_count > 0:
        print(f"Successfully excluded {excluded_count} file(s)")
    else:
        print("Note: No files were excluded.")

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Combine files with improved structure
    with open(output_file, "w", encoding="utf-8") as outfile:
        # Add a comprehensive header
        outfile.write("# Cobaya Documentation\n\n")
        outfile.write("---\n\n")

        # Add each file's content
        for file_path in filtered_files:
            file_name = os.path.basename(file_path)
            section_name = os.path.splitext(file_name)[0]

            print(f"  Adding {section_name}...")
            outfile.write(f"## {file_name}\n\n")

            # Add file content
            with open(file_path, encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write("\n\n")

    print(f"Combined documentation written to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    return True


def main():
    args = parse_args()

    try:
        # Get the list of files to exclude
        exclude_files = args.exclude.split(",") if args.exclude else []

        # Install dependencies if not skipped
        if not args.no_install and not install_dependencies():
            print("Failed to install required dependencies. Exiting.")
            return 1

        # Update conf.py
        if not update_conf_py():
            print("Failed to update conf.py. Exiting.")
            return 1

        # Build the documentation
        build_dir = build_markdown_docs()

        # Combine the files
        if not combine_markdown_files(build_dir, exclude_files, args.output):
            print("Failed to combine markdown files. Exiting.")
            return 1

        # Verify the file exists and has content
        if not os.path.exists(args.output):
            print(f"ERROR: Failed to generate markdown file at {args.output}")
            return 1

        file_size = os.path.getsize(args.output)
        print(f"Final markdown file size: {file_size / 1024:.2f} KB")

        if file_size == 0:
            print("ERROR: Generated markdown file is empty")
            return 1

        print(f"\nSuccess! Documentation has been built and combined into: {args.output}")
        print("This file can now be used as context for Large Language Models.")
        return 0

    except Exception as e:
        print(f"ERROR: An exception occurred during the build process: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
