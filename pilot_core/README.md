# pilot_core

Rust core of the residual-RL racing pilot, exposed to Python with PyO3 (built by maturin). It runs the
per-physics-step part of `pilot.closedloop.LapSim` when a config sets `kernel: rust`; Python crosses into Rust
once per control interval (20 Hz) and keeps the MPC, the residual policy, logging and the summary.

| Module | Mirrors |
|---|---|
| `dynamics.rs` | Gym-Khana 1.2.0 `vehicle_dynamics_st`, `pilot.tires` (Pacejka / Fiala + friction circle), RK4, `RaceCar.update_pose` (2-step steering FIFO, yaw wrap) |
| `geometry.rs` | `TrackGeometry.check`: tire patches vs the legal region, body vs wall cells (exact `orient2d`) |
| `track.rs` | `Reference.project`, the periodic speed-plan lookup |
| `kernel.rs` | `LapSim._physics`, `_pre`, `advance_interval` (servo, speed controller, metrics, wall / spin / timeout) |
| `numerics.rs` | Python / NumPy float semantics (see below) |

It does not depend on Gym-Khana: track, reference line, speed plan and parameters come in as arrays and a
params dict (`pilot/kernel.py` builds them).

## Bit-identical to the Python loop

Laps with `kernel: rust` are bit-identical to `kernel: python` (`uv run python -m pilot.kernel_check`). On
macOS arm64 that needs:

* scalar `x ** 2` → libm `pow` (not `x * x`; they differ for some inputs), exponent behind `black_box`;
* `np.interp` → `mul_add` (NumPy's interp is compiled with FMA contraction);
* every sine through `numerics::sin` (`black_box` argument): LLVM would otherwise fuse a `sin`/`cos` pair into
  Darwin's `__sincos_stret`, whose sine differs from libm `sin` for ~0.2% of arguments;
* Python modulo / `max` / `min` and `np.clip` comparison orders; expressions in Python's evaluation order.

Other platforms (a different libm) may need the same checks repeated.

## Build

Needs a Rust toolchain (`rust-toolchain.toml` pins 1.100). `uv sync` in the repository root builds and installs
the extension (it is a path dependency); it rebuilds when the Rust sources change. With Homebrew's keg-only
`rustup`, put it on `PATH` first:

```bash
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
```

Rust unit tests: `cargo test` in this directory (set `PYO3_PYTHON` to the project's `.venv/bin/python`).
