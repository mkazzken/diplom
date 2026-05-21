# Stefan Problem Streamlit Sandbox

This sandbox provides an interactive Streamlit interface for the 2D one-phase Stefan problem.

## Run instructions

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## Notes

- `main.py` is left untouched and remains the authoritative reference implementation.
- The sandbox uses the same enthalpy formulation, boundary conditions, heat source model, and Stefan condition as `main.py`.
- Use the sidebar to configure materials, heat source settings, mesh size, time step, and simulation parameters.
