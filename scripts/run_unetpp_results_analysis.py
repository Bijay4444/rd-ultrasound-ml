"""Execute the read-only reporting notebook and save its outputs outside Git.

Run with uv and temporary reporting dependencies; see docs/UNETPP_RESULTS.md.
"""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs/unetpp_analysis'
    output.mkdir(parents=True, exist_ok=True)
    for name, suffix in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython'),
                         ('JUPYTER_RUNTIME_DIR', 'jupyter')]:
        os.environ.setdefault(name, str(output / suffix))
    notebook = nbformat.read(root / 'notebooks/unetpp_results_analysis.ipynb', as_version=4)
    NotebookClient(notebook, timeout=180, kernel_name='python3',
        resources={'metadata': {'path': str(root)}}).execute()
    nbformat.validate(notebook)
    destination = output / 'unetpp_results_analysis_executed.ipynb'
    nbformat.write(notebook, destination)
    print(f'Executed all code cells successfully: {destination}')


if __name__ == '__main__':
    main()
