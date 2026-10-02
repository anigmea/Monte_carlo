"""Seeded regression for the app's daily-log-return unit contract.
Run: python test_gbm_moments.py (requires numpy only, no market/API access).
"""
import ast
from pathlib import Path
import numpy as np

source = ast.parse(Path(__file__).with_name("app.py").read_text())
function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "simulate_gbm_paths")
namespace = {"np": np}
exec(compile(ast.Module(body=[function], type_ignores=[]), "app.py", "exec"), namespace)
simulate = namespace["simulate_gbm_paths"]
np.random.seed(20261001)
mu = np.array([0.0004, 0.0002])
vol = np.array([0.02, 0.03])
cov = np.array([[1.0, 0.6], [0.6, 1.0]]) * np.outer(vol, vol)
for dt in (1.0, 5.0):
    prices = simulate(mu, cov, [100.0, 100.0], n_steps=1, n_sims=150_000, dt=dt)
    returns = np.log(prices[:, 0, :] / 100.0)
    np.testing.assert_allclose(returns.std(axis=0), vol * np.sqrt(dt), rtol=0.01)
    assert abs(np.corrcoef(returns.T)[0, 1] - 0.6) < 0.01
    assert np.all(np.abs(returns.mean(axis=0) - mu * dt) < 6 * vol * np.sqrt(dt) / np.sqrt(len(returns)))
    print(f"dt={dt}: means={returns.mean(axis=0)}, vols={returns.std(axis=0)}, correlation={np.corrcoef(returns.T)[0, 1]:.6f}")
print("PASS: daily and five-day log-return moments and correlation")
