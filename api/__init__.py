"""API boundary for the Focus Intelligence Engine.

This package is deliberately **outside** ``src/focus_engine``. The engine is a library of
layers with a fixed dependency direction, and its root package intentionally exposes no
layer symbols so that a layer can never be reached without its declared dependencies.
Adding a transport concern inside that package would either violate that rule or require
weakening the layer-boundary tests that enforce it.

Nothing in ``src/focus_engine`` imports this package, and this package imports only the
engine's public surfaces. The dependency direction is therefore one-way:

    EduFlow / lab UI  ->  api  ->  focus_engine  ->  (no path back)

The lab under :mod:`api.lab` is a **development-only** interface. It exists so a human can
watch the real engine process real-shaped data and inspect why it decided what it decided.
It is not the production EduFlow UI, and it makes no claim that any number it displays is
accurate: every corpus available to it is synthetic simulator output.
"""
