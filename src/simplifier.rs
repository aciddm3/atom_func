use crate::func::{EPS, EnvFunction};

impl EnvFunction {
    #[inline]
    pub fn simplify<const MAX_RECURSION_DEPTH: usize>(&mut self) {
        self.simplify_with_depth::<MAX_RECURSION_DEPTH>(0);
    }

    fn simplify_with_depth<const MAX_RECURSION_DEPTH: usize>(&mut self, depth: usize) {
        if depth > MAX_RECURSION_DEPTH {
            return;
        } //  stack overflow protection

        match self {
            Self::Id(inner)
            | Self::Exp(inner)
            | Self::Sigmoid(inner)
            | Self::Ln(inner)
            | Self::Sin(inner)
            | Self::Cos(inner)
            | Self::InvVal(inner)
            | Self::Neg(inner)
            | Self::Abs(inner)
            | Self::Linear(inner, _, _)
            | Self::Powf(inner, _)
            | Self::Powi(inner, _)
            | Self::Periodic(inner, _)
            | Self::ConstMul(inner, _) => {
                inner.simplify_with_depth::<MAX_RECURSION_DEPTH>(depth + 1);
            }
            Self::Sum(a, b) | Self::Dif(a, b) | Self::Mul(a, b) | Self::Div(a, b) => {
                a.simplify_with_depth::<MAX_RECURSION_DEPTH>(depth + 1);
                b.simplify_with_depth::<MAX_RECURSION_DEPTH>(depth + 1);
            }
            _ => {}
        }
        self.apply_simp_rules();
    }

    fn try_fold_constants(&self) -> Option<f32> {
        match self {
            // Унарные: если аргумент уже Constant → всё дерево константно
            Self::Exp(inner)
            | Self::Sigmoid(inner)
            | Self::Ln(inner)
            | Self::Sin(inner)
            | Self::Cos(inner)
            | Self::Neg(inner)
            | Self::Abs(inner)
            | Self::Linear(inner, _, _)
                if matches!(inner.as_ref(), Self::Constant(_)) =>
            {
                return Some(self.eval(0.0)); // eval сам применит защиту от NaN/Inf
            }
            Self::Dif(l, r) | Self::Mul(l, r) | Self::Div(l, r)
                if matches!(l.as_ref(), Self::Constant(_))
                    && matches!(r.as_ref(), Self::Constant(_)) =>
            {
                return Some(self.eval(0.0));
            }
            _ => {}
        }
        None
    }

    #[inline]
    fn apply_simp_rules(&mut self) {
        if let Some(val) = self.try_fold_constants() {
            *self = Self::Constant(val);
            return;
        }

        match self {
            Self::Id(inner) => {
                *self = *inner.clone();
            }
            Self::ConstMul(inner, c) => {
                if *c == 0.0 {
                    *self = Self::Constant(0.0);
                    return;
                }
                if *c == 1.0 {
                    *self = *inner.clone();
                    return;
                }
                if *c == -1.0 {
                    *self = Self::Neg(inner.clone());
                    return;
                }
                if let Self::ConstMul(inner2, b) = inner.as_mut() {
                    *self = Self::ConstMul(inner2.clone(), *b * *c);
                }
            }

            Self::Exp(_inner) => {}

            Self::Sigmoid(_inner) => {}

            Self::Ln(_inner) => {}

            Self::Sin(_inner) => {}

            Self::Cos(inner) => {
                if let Self::Abs(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                } else if let Self::Neg(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                }
            }

            Self::Linear(inner, k, b) => {
                if *k == 0.0 {
                    *self = Self::Constant(*b);
                    return;
                }
                if *b == 0.0 {
                    *self = Self::ConstMul(inner.clone(), *k);
                }
            }

            Self::Powf(inner, p) => {
                if *p == 0.0 {
                    *self = Self::Constant(1.0);
                } else if *p == 1.0 {
                    *self = *inner.clone();
                } else if let Self::Powf(v, p2) = inner.as_mut() {
                    *self = Self::Powf(v.clone(), *p * *p2);
                } else if let Self::Powi(v, p2) = inner.as_mut() {
                    *self = Self::Powf(v.clone(), *p * *p2 as f32);
                }
            }

            Self::Powi(inner, p) => {
                if let Self::Powf(v, p2) = inner.as_ref() {
                    *self = Self::Powf(v.clone(), *p2 * *p as f32);
                } else if let Self::Powi(v, p2) = inner.as_ref() {
                    *self = Self::Powi(v.clone(), *p2 * *p);
                } else if *p == 0 {
                    *self = Self::Constant(1.0);
                } else if *p == 1 {
                    *self = *inner.clone();
                } else if *p % 2 == 0 {
                    if let Self::Abs(inner2) = inner.as_mut() {
                        *inner = inner2.clone();
                    }
                    if let Self::Neg(inner2) = inner.as_mut() {
                        *inner = inner2.clone();
                    }
                }
            }

            Self::Periodic(_inner, _) => {}

            Self::InvVal(inner) => {
                if let Self::InvVal(inner2) = inner.as_mut() {
                    *self = *inner2.clone();
                } else if let Self::Powf(inner2, p) = inner.as_ref() {
                    *self = Self::Powf(inner2.clone(), -*p);
                } else if let Self::Powi(inner2, p) = inner.as_ref() {
                    *self = Self::Powi(inner2.clone(), -*p);
                }
            }

            Self::Neg(inner) => {
                if let Self::Neg(inner2) = inner.as_mut() {
                    *self = *inner2.clone();
                } else if let Self::ConstMul(inner2, c) = inner.as_ref() {
                    *self = Self::ConstMul(inner2.clone(), -*c); // -(c*x) → (-c)*x
                }
            }

            Self::Abs(inner) => {
                if let Self::Abs(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                } else if let Self::Neg(inner2) = inner.as_mut() {
                    *inner = inner2.clone();
                }
            }

            Self::Sum(l, r) => {
                if let Self::Constant(v) = l.as_ref() {
                    *self = Self::Linear(r.clone(), 1.0, *v);
                    return;
                }
                if let Self::Constant(v) = r.as_ref() {
                    *self = Self::Linear(l.clone(), 1.0, *v);
                    return;
                }
                if let Self::Neg(inner_l) = l.as_ref()
                    && let Self::Neg(inner_r) = r.as_ref()
                {
                    *self = Self::Neg(Box::new(Self::Sum(inner_l.clone(), inner_r.clone())))
                }
            }

            Self::Dif(l, r) => {
                if let Self::Constant(v) = l.as_ref() {
                    *self = Self::Linear(r.clone(), -1.0, *v);
                    return;
                }
                if let Self::Constant(v) = r.as_ref() {
                    *self = Self::Linear(l.clone(), 1.0, -*v);
                    return;
                }
                if let Self::Neg(inner_l) = l.as_ref()
                    && let Self::Neg(inner_r) = r.as_ref()
                {
                    *self = Self::Neg(Box::new(Self::Dif(inner_l.clone(), inner_r.clone())))
                }
            }

            Self::Mul(l, r) => {
                if let Self::Constant(c) = l.as_ref() {
                    *self = Self::ConstMul(r.clone(), *c);
                    return;
                }
                if let Self::Constant(c) = r.as_ref() {
                    *self = Self::ConstMul(l.clone(), *c);
                    return;
                }
                if let Self::Neg(inner_l) = l.as_mut()
                    && let Self::Neg(inner_r) = r.as_mut()
                {
                    *l = inner_l.clone();
                    *r = inner_r.clone();
                }
                if let Self::Abs(inner_l) = l.as_ref()
                    && let Self::Abs(inner_r) = r.as_ref()
                {
                    *self = Self::Abs(Box::new(Self::Mul(inner_l.clone(), inner_r.clone())))
                }
            }

            Self::Div(l, r) => {
                if let Self::Constant(c) = r.as_ref() {
                    if c.abs() < EPS {
                        *self = Self::Constant(0.0);
                    } else {
                        *self = Self::ConstMul(l.clone(), 1.0 / *c);
                    }
                    return;
                }
                if let Self::Neg(inner_l) = l.as_mut()
                    && let Self::Neg(inner_r) = r.as_mut()
                {
                    *l = inner_l.clone();
                    *r = inner_r.clone();
                }
                if let Self::Abs(inner_l) = l.as_ref()
                    && let Self::Abs(inner_r) = r.as_ref()
                {
                    *self = Self::Abs(Box::new(Self::Div(inner_l.clone(), inner_r.clone())))
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod test {
    use crate::func::EnvFunction;
    fn assert_eval_eq(f: &EnvFunction, x: f32, expected: f32, tolerance: f32) {
        let actual = f.eval(x);
        assert!(
            (actual - expected).abs() < tolerance,
            "f({}) = {}, expected {}",
            x,
            actual,
            expected
        );
    }

    #[test]
    fn test_simplify_id_elimination() {
        let mut f = EnvFunction::Id(Box::new(EnvFunction::Constant(42.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 42.0, 1e-5);
    }

    #[test]
    fn test_simplify_constant_folding_unary() {
        // Exp(Constant(0)) → Constant(1)
        let mut f = EnvFunction::Exp(Box::new(EnvFunction::Constant(0.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 123.45, 1.0, 1e-5); // x не важен, функция константна

        // Sin(Constant(π/2)) ≈ 1
        let mut f = EnvFunction::Sin(Box::new(EnvFunction::Constant(std::f32::consts::FRAC_PI_2)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 1.0, 1e-4);
    }

    #[test]
    fn test_simplify_const_mul() {
        // 0 * f(x) → 0
        let mut f =
            EnvFunction::ConstMul(Box::new(EnvFunction::Sin(Box::new(EnvFunction::Arg))), 0.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 1.0, 0.0, 1e-5);

        // 1 * f(x) → f(x)
        let mut f = EnvFunction::ConstMul(Box::new(EnvFunction::Arg), 1.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 5.0, 5.0, 1e-5);

        // c * Constant(v) → Constant(c*v)
        let mut f = EnvFunction::ConstMul(Box::new(EnvFunction::Constant(3.0)), 4.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 12.0, 1e-5);
    }

    #[test]
    fn test_simplify_linear() {
        // k=0: Linear(f, 0, b) → Constant(b)
        let mut f = EnvFunction::Linear(Box::new(EnvFunction::Arg), 0.0, 7.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 100.0, 7.0, 1e-5);

        // b=0: Linear(f, k, 0) → ConstMul(f, k)
        let mut f = EnvFunction::Linear(Box::new(EnvFunction::Arg), 2.0, 0.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 3.0, 6.0, 1e-5);
    }

    #[test]
    fn test_simplify_double_neg() {
        // Neg(Neg(x)) → x
        let mut f = EnvFunction::Neg(Box::new(EnvFunction::Neg(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, -5.0, -5.0, 1e-5);
        assert_eval_eq(&f, 3.14, 3.14, 1e-5);
    }

    #[test]
    fn test_simplify_abs_abs() {
        // Abs(Abs(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(EnvFunction::Abs(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, -7.0, 7.0, 1e-5);
        assert_eval_eq(&f, 2.0, 2.0, 1e-5);
    }

    #[test]
    fn test_simplify_abs_neg() {
        // Abs(Neg(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(EnvFunction::Neg(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, -4.0, 4.0, 1e-5);
    }

    #[test]
    fn test_simplify_inv_inv() {
        // InvVal(InvVal(x)) → x (для x ≠ 0)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::InvVal(Box::new(EnvFunction::Arg))));
        f.simplify::<128>();
        assert_eval_eq(&f, 2.0, 2.0, 1e-5);
        assert_eval_eq(&f, -3.0, -3.0, 1e-5);
        // Для 0: защита от деления на ноль → 0
        assert_eval_eq(&f, 0.0, 0.0, 1e-5);
    }

    // ─────────────────────────────────────────────────────
    // ➕ Суммы и разности с константами
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_sum_constant_left() {
        // Constant(c) + f(x) → Linear(f, 1, c)
        let mut f = EnvFunction::Sum(
            Box::new(EnvFunction::Constant(5.0)),
            Box::new(EnvFunction::Arg),
        );
        f.simplify::<128>();
        assert_eval_eq(&f, 10.0, 15.0, 1e-5); // 10 + 5
    }

    #[test]
    fn test_simplify_sum_constant_right() {
        // f(x) + Constant(c) → Linear(f, 1, c)
        let mut f = EnvFunction::Sum(
            Box::new(EnvFunction::Arg),
            Box::new(EnvFunction::Constant(-3.0)),
        );
        f.simplify::<128>();
        assert_eval_eq(&f, 10.0, 7.0, 1e-5); // 10 - 3
    }

    #[test]
    fn test_simplify_dif_constant() {
        // Constant(c) - f(x) → Linear(f, -1, c)
        let mut f = EnvFunction::Dif(
            Box::new(EnvFunction::Constant(10.0)),
            Box::new(EnvFunction::Arg),
        );
        f.simplify::<128>();
        assert_eval_eq(&f, 4.0, 6.0, 1e-5); // 10 - 4

        // f(x) - Constant(c) → Linear(f, 1, -c)
        let mut f2 = EnvFunction::Dif(
            Box::new(EnvFunction::Arg),
            Box::new(EnvFunction::Constant(2.0)),
        );
        f2.simplify::<128>();
        assert_eval_eq(&f2, 5.0, 3.0, 1e-5); // 5 - 2
    }

    #[test]
    fn test_simplify_periodic_constant() {
        // Periodic(Constant(v), p) → Constant(v)
        let mut f = EnvFunction::Periodic(Box::new(EnvFunction::Constant(1.5)), 2.0);
        f.simplify::<128>();
        assert_eval_eq(&f, 100.0, 1.5, 1e-5);
    }

    #[test]
    fn test_simplify_nested_simplify() {
        // Exp(Sin(Constant(0))) → Exp(0) → 1
        let mut f = EnvFunction::Exp(Box::new(EnvFunction::Sin(Box::new(EnvFunction::Constant(
            0.0,
        )))));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 1.0, 1e-3);
    }

    // ─────────────────────────────────────────────────────
    // 🛡 Защита от переполнения стека
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_depth_limit() {
        // Создаём дерево глубины 50: Id(Id(Id(...(Arg)...)))
        let mut deep = EnvFunction::Arg;
        for _ in 0..50 {
            deep = EnvFunction::Id(Box::new(deep));
        }

        // Не должно паниковать — защита на 32 уровнях
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            deep.simplify::<128>();
        }));

        assert!(result.is_ok(), "simplify паникнул на глубоком дереве");
    }

    #[test]
    fn test_simplify_depth_limit_nested_ops() {
        // Глубокое дерево с операциями: Exp(Exp(Exp(...)))
        let mut deep = EnvFunction::Arg;
        for _ in 0..40 {
            deep = EnvFunction::Exp(Box::new(deep));
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            deep.simplify::<128>();
        }));

        assert!(result.is_ok(), "simplify паникнул на глубоком Exp-дереве");
    }

    // ─────────────────────────────────────────────────────
    // 🔒 Безопасность eval после simplify
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_nan_safety() {
        // Ln(Constant(-1)) → NaN → 0 (защита в eval)
        let mut f = EnvFunction::Ln(Box::new(EnvFunction::Constant(-1.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 0.0, 1e-5); // NaN превращается в 0
    }

    #[test]
    fn test_simplify_div_zero_safety() {
        // InvVal(Constant(0)) → 0 (защита в eval)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::Constant(0.0)));
        f.simplify::<128>();
        assert_eval_eq(&f, 0.0, 0.0, 1e-5);
    }

    // ─────────────────────────────────────────────────────
    // 🎯 Интеграционный тест: сложное выражение
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_complex_expression() {
        // f(x) = 2 * (Sin(x) + 3) - 1
        // После simplify: Linear(ConstMul(Sin(Arg), 2), 1, 5) или эквивалент
        let f = EnvFunction::Dif(
            Box::new(EnvFunction::ConstMul(
                Box::new(EnvFunction::Sum(
                    Box::new(EnvFunction::Sin(Box::new(EnvFunction::Arg))),
                    Box::new(EnvFunction::Constant(3.0)),
                )),
                2.0,
            )),
            Box::new(EnvFunction::Constant(1.0)),
        );

        let mut f_simplified = f.clone();
        f_simplified.simplify::<128>();

        // Проверяем, что значения совпадают на нескольких точках
        let points = vec![0.0, 1.0, std::f32::consts::PI / 2.0, std::f32::consts::PI];
        for x in points {
            let original = f.eval(x);
            let simplified = f_simplified.eval(x);
            assert!(
                (original - simplified).abs() < 1e-4,
                "Расхождение при x={}: {} vs {}",
                x,
                original,
                simplified
            );
        }
    }
    #[test]
    fn test_pow_composition() {
        // (x^2)^3 → x^6
        let mut f = EnvFunction::Powf(
            Box::new(EnvFunction::Powi(Box::new(EnvFunction::Arg), 2)),
            3.0,
        );
        f.simplify::<64>();

        if let EnvFunction::Powf(inner, p) = &f {
            assert!(matches!(**inner, EnvFunction::Arg));
            assert!((p - 6.0).abs() < 1e-5);
        } else {
            panic!("Expected Powf(Arg, 6.0)");
        }
    }

    #[test]
    fn test_invval_pow() {
        // 1/(x^2) → x^(-2)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::Powi(Box::new(EnvFunction::Arg), 2)));
        f.simplify::<64>();

        if let EnvFunction::Powi(inner, p) = &f {
            assert!(matches!(**inner, EnvFunction::Arg));
            assert_eq!(*p, -2);
        } else {
            panic!("Expected Powi(Arg, -2)");
        }
    }
}
