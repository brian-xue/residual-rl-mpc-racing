//! Scalar helpers that reproduce Python / NumPy floating-point semantics bit for bit.
//!
//! The kernel is checked against the Python implementation for bit-identical trajectories, so every
//! operation here mirrors what CPython / NumPy actually execute on this platform (verified on macOS arm64):
//! * scalar `x ** 2` (Python float or NumPy scalar) calls libm `pow`, which is not always equal to `x * x`;
//! * `np.interp` is compiled with FMA contraction: `slope * (x - xp[j]) + fp[j]` is one fused multiply-add;
//! * `%` on floats is Python's modulo (sign of the divisor), `np.clip` / built-in `max` / `min` have their own
//!   comparison orders (which matter for NaN and signed zeros).
//!
//! Rust never contracts `a * b + c` on its own, and `sin`, `cos`, `tan`, `atan`, `hypot` go to the same libm
//! (sines through [`sin`], which blocks the sin/cos fusion LLVM does on macOS).

unsafe extern "C" {
    fn pow(x: f64, y: f64) -> f64;
}

/// Scalar `x ** 2` as CPython / NumPy scalars compute it (libm `pow`). The exponent goes through
/// `black_box` so LLVM cannot rewrite the call into `x * x`.
#[inline]
pub fn pow2(x: f64) -> f64 {
    unsafe { pow(x, std::hint::black_box(2.0)) }
}

/// libm `sin`. The argument goes through `black_box` so LLVM cannot fuse a `sin` / `cos` pair of the same
/// argument into Darwin's `__sincos_stret`, whose sine differs from libm `sin` in ~0.2% of arguments
/// (its cosine matches). Use this for every sine.
#[inline]
pub fn sin(x: f64) -> f64 {
    std::hint::black_box(x).sin()
}

/// Python / NumPy float modulo: result has the sign of the divisor, `-0.0` results become `+0.0 * sign(b)`.
#[inline]
pub fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b; // fmod
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) { m + b } else { m }
    } else {
        0.0f64.copysign(b)
    }
}

/// Python integer modulo for a positive divisor.
#[inline]
pub fn py_imod(a: i64, n: i64) -> i64 {
    a.rem_euclid(n)
}

/// Built-in `max(a, b)`: returns `a` unless `b > a`.
#[inline]
pub fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Built-in `min(a, b)`: returns `a` unless `b < a`.
#[inline]
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

/// `np.clip(x, lo, hi)` for float64 scalars: `_NPY_MIN(_NPY_MAX(x, lo), hi)` with NaN propagation.
#[inline]
#[allow(clippy::if_same_then_else)] // the NaN branches spell out NumPy's _NPY_MAX / _NPY_MIN
pub fn np_clip(x: f64, lo: f64, hi: f64) -> f64 {
    let m = if x.is_nan() {
        x
    } else if x > lo {
        x
    } else {
        lo
    };
    if m.is_nan() {
        m
    } else if m < hi {
        m
    } else {
        hi
    }
}

/// `np.interp(x, xp, fp)` for a scalar `x` and strictly increasing `xp` (left = fp[0], right = fp[-1]).
pub fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x.is_nan() {
        return x;
    }
    if x > xp[n - 1] {
        return fp[n - 1];
    }
    if x < xp[0] {
        return fp[0];
    }
    // j with xp[j] <= x < xp[j + 1] (or j = n - 1 when x == xp[n - 1])
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let mut r = slope.mul_add(x - xp[j], fp[j]);
    if r.is_nan() {
        r = slope.mul_add(x - xp[j + 1], fp[j + 1]);
        if r.is_nan() && fp[j] == fp[j + 1] {
            r = fp[j];
        }
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulo_follows_python() {
        assert_eq!(py_mod(-1.0, 3.0), 2.0);
        assert_eq!(py_mod(7.5, 2.0), 1.5);
        assert!(py_mod(-0.0, 2.0).is_sign_positive());
        assert_eq!(py_imod(-3, 10), 7);
    }

    #[test]
    fn interp_edges() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 10.0, 30.0];
        assert_eq!(np_interp(-1.0, &xp, &fp), 0.0);
        assert_eq!(np_interp(2.0, &xp, &fp), 30.0);
        assert_eq!(np_interp(3.0, &xp, &fp), 30.0);
        assert_eq!(np_interp(1.0, &xp, &fp), 10.0);
        assert_eq!(np_interp(1.5, &xp, &fp), 20.0);
    }
}
