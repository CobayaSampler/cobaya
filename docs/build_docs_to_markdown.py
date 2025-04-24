"""
Script to build Sphinx documentation and combine it into a single Markdown file.
This script can be used in a GitHub Actions workflow or locally.

Usage:
    python build_docs_to_markdown.py [--exclude file1,file2,...] [--output output_file]

Options:
    --exclude: Comma-separated list of files to exclude (without .md extension)
    --output: Output file path (default: docs/_build/cobaya_docs_combined.md)
"""

import os
import sys
import subprocess
import argparse
import glob


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Sphinx documentation and combine it into a single Markdown file."
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help="Comma-separated list of files to exclude (without .md extension)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="docs/_build/cobaya_docs_combined.md",
        help="Output file path",
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
    """Update conf.py to include sphinx_markdown_builder extension."""
    conf_path = "docs/conf.py"
    with open(conf_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if sphinx_markdown_builder is already in the file
    if "sphinx_markdown_builder" in content:
        print("sphinx_markdown_builder is already in conf.py")
        return True

    print("Adding sphinx_markdown_builder to conf.py...")

    # First, try to find the try-except block for optional extensions
    if "# Add sphinx_markdown_builder if available" in content:
        # The structure for optional extensions is already there
        # We don't need to do anything
        print("Optional extension structure already exists in conf.py")
        return True

    # If not found, check if we can add it to the extensions list directly
    if "extensions = [" in content and "]" in content:
        # Find the end of the extensions list
        extensions_start = content.find("extensions = [")
        extensions_end = content.find("]", extensions_start)

        if extensions_end > extensions_start:
            # Insert the try-except block after the extensions list
            new_content = (
                    content[:extensions_end + 1] +
                    "\n\n# Add sphinx_markdown_builder if available (used for LLM context generation)" +
                    "\ntry:" +
                    "\n    import sphinx_markdown_builder" +
                    "\n    extensions.append('sphinx_markdown_builder')" +
                    "\nexcept ImportError:" +
                    "\n    pass" +
                    content[extensions_end + 1:]
            )

            # Write the updated content back to the file
            with open(conf_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return True

    print(
        "Warning: Could not find a suitable place to add sphinx_markdown_builder in conf.py")
    return False


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

    return build_dir


def combine_markdown_files(build_dir, exclude_files, output_file):
    """Combine Markdown files into a single file."""
    print(f"Combining Markdown files into {output_file}...")

    # Get all markdown files
    md_files = sorted(glob.glob(os.path.join(build_dir, "*.md")))

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
        print("Warning: No files were excluded. Check your exclude patterns.")

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Combine files with structure
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("# Cobaya Documentation\n\n")

        for file_path in filtered_files:
            file_name = os.path.basename(file_path)
            section_name = os.path.splitext(file_name)[0]

            print(f"  Adding {section_name}...")

            # Add section header
            outfile.write(f"## {file_name}\n\n")

            # Add file content
            with open(file_path, "r", encoding="utf-8") as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write("\n\n---\n\n")

    print(f"Combined documentation written to {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")


def main():
    """Main function."""
    args = parse_args()

    # Get the list of files to exclude
    exclude_files = args.exclude.split(",") if args.exclude else []

    # Install dependencies
    if not install_dependencies():
        print("Failed to install required dependencies. Exiting.")
        return 1

    # Update conf.py
    if not update_conf_py():
        print("Failed to update conf.py. Exiting.")
        return 1

    # Build the documentation
    build_dir = build_markdown_docs()

    # Combine the files
    combine_markdown_files(build_dir, exclude_files, args.output)

    print(f"\nSuccess! Documentation has been built and combined into: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
