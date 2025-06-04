.. _llm_context:

Documentation for LLM Context
=============================

Cobaya's documentation is available in a single Markdown file format, specifically designed for use as context for Large Language Models (LLMs).

Why Markdown for LLMs?
----------------------

Markdown is a lightweight markup language that is easy for both humans and machines to read. It's particularly well-suited for LLMs because:

1. It preserves the structure of the documentation
2. It's plain text, making it easy to process
3. It's compact, allowing more content to fit within token limits

Downloading the Markdown Documentation
---------------------------------------

You can download the complete Cobaya documentation as a single Markdown file:

* `Download as Markdown <_static/cobaya_docs_combined.md>`_ - Plain text format for direct use with LLMs

This file contains the entire documentation, including all sections, code examples, and explanations.

Using the Markdown Documentation with LLMs
------------------------------------------

The Markdown documentation can be used as context for LLMs in several ways:

1. **Direct upload**: Many LLM platforms allow you to upload documents as context into a custom agent system prompt
2. **Embedding**: The documentation can be embedded and used with retrieval-augmented generation (RAG)
3. **Copy-paste**: For one-off use, you can copy and paste directly into your prompts

The documentation is structured to maintain the same organization as the HTML version, making it easy to find specific information.

Generating the Markdown Documentation Locally
--------------------------------------------

If you want to generate the markdown documentation locally, you can use the provided script:

.. code-block:: bash

    python docs/markdown_builder.py

This will:

1. Install the required dependencies
2. Build the Sphinx documentation in markdown format
3. Combine all markdown files into a single file at ``docs/cobaya_docs_combined.md``

You can customize the output with these options:

.. code-block:: bash

    python docs/markdown_builder.py --exclude "file1,file2" --output "custom_path.md" --no-install

Where:

* ``--exclude``: Comma-separated list of files to exclude (without .md extension)
* ``--output``: Custom output file path
* ``--no-install``: Skip installation of dependencies
