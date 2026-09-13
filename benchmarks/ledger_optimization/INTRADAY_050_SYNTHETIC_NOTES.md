# Before-freeze numerical checks

No development outcomes were computed while correcting these synthetic failures.
The fixture has six constant model predictions1 through6 and four six-hour
target blocks alternating2 and5. The unit tests for analytic gradients, block
assignment and invalid inputs passed throughout.

The initial exact-norm objective fit this synthetic fixture almost exactly,
but its conservative omitted-near-zero gradient certificate had gap0.00569410
despite SLSQP success (70 iterations, objective0.00123286146). That certificate
was too loose to accept the fit. It was not called a successful numerical test.

Replacing the norm by sqrt(MSLE+1e-12) bounds objective perturbation by1e-6 and
retains a defined gradient. SLSQP at ftol1e-12 still stopped with gap2.68607e-5
after54 iterations; tightening ftol alone to1e-14 did not pass the1e-5 gate.
Neither failed test relaxed that gate.

Added at most32 recorded convex line refinements on the same smooth objective,
with first-order gap acceptance unchanged. All four tests then passed, including
the exact-fit fixture. Freeze this final objective and bounded solver before
the416-case development experiment. Raw command failures remain in the session
tool history; this note preserves their cause and correction in the repository.
