```md
# Applied Bioinformatics Case Study

This repository contains the code, data processing pipeline, and
exploratory analysis for the Applied Bioinformatics coursework case
study (2026).

The project is structured as a small Python package using a `src/`
layout, with data loading and preprocessing logic separated from
notebooks and analysis code.

---

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
````

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Install the project package (editable mode)

```bash
pip install -e .
```

This makes the `applied` package importable in notebooks and scripts.

---

## Usage

After installation, project modules can be imported as:

```python
from applied.data_processing import build_features_and_target
```

Raw data are loaded from the `data/` directory, and notebooks are
intended to be run from the `notebooks/` folder.

---

## Notes

* Tested with Python 3.10+
* The `src/` layout is used to avoid import ambiguity and ensure clean
  separation between source code and analysis notebooks.
* Parallel sensor channels (e.g. liquid inflow streams) are aggregated
  based on physical interpretation, as described in the accompanying
  analysis.

---

## Author

Applied Bioinformatics Case Study
2026


```
