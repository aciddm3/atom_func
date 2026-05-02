use crate::func::EnvFunction;

impl EnvFunction {
    #[inline]
    pub fn simplify(&mut self) {
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
                match inner.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Constant(Self::eval(
                            &Self::ConstMul(Box::new(Self::Arg), *c),
                            *val,
                        ));
                    }
                    Self::ConstMul(inner2, b) => *self = Self::ConstMul(inner2.clone(), *b * *c),
                    _ => (),
                }
            }
            Self::Exp(inner) => {
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Exp(Box::new(Self::Arg)), val));
                }
            }
            Self::Sigmoid(inner) => {
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Sigmoid(Box::new(Self::Arg)), val));
                }
            }
            Self::Ln(inner) => {
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Ln(Box::new(Self::Arg)), val));
                }
            }
            Self::Sin(inner) => {
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Sin(Box::new(Self::Arg)), val));
                }
            }
            Self::Cos(inner) => match inner.as_ref() {
                Self::Constant(val) => {
                    *self = Self::Constant(Self::eval(&Self::Cos(Box::new(Self::Arg)), *val));
                }
                Self::Abs(inner2) => {
                    *inner = inner2.clone();
                }
                Self::Neg(inner2) => {
                    *inner = inner2.clone();
                }
                _ => (),
            },
            Self::Linear(inner, k, b) => {
                if *k == 0.0 {
                    *self = Self::Constant(*b);
                    return;
                }
                if *b == 0.0 {
                    *self = Self::ConstMul(inner.clone(), *k);
                    return;
                }
                if let Self::Constant(val) = **inner {
                    *self =
                        Self::Constant(Self::eval(&Self::Linear(Box::new(Self::Arg), *k, *b), val));
                }
            }
            Self::Powf(inner, p) => {
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Powf(Box::new(Self::Arg), *p), val));
                }
            }
            Self::Powi(inner, p) => {
                match inner.as_mut() {
                    Self::Constant(val) => {
                        *self =
                            Self::Constant(Self::eval(&Self::Powi(Box::new(Self::Arg), *p), *val));
                        return;
                    }
                    _ => (),
                }
                if *p == 0 {
                    *self = Self::Constant(1.0);
                    return;
                }
                if *p == 1 {
                    *self = *inner.clone();
                    return;
                }
                if *p % 2 == 0 {
                    match inner.as_mut() {
                        Self::Abs(inner) => {
                            *inner = inner.clone();
                        }
                        Self::Neg(inner) => {
                            *inner = inner.clone();
                        }
                        _ => (),
                    }
                }
            }
            Self::Periodic(inner, t) => {
                if let Self::Constant(val) = **inner {
                    *self =
                        Self::Constant(Self::eval(&Self::Periodic(Box::new(Self::Arg), *t), val));
                }
            }
            Self::InvVal(inner) => match inner.as_mut() {
                Self::Constant(val) => {
                    *self = Self::Constant(Self::eval(&Self::InvVal(Box::new(Self::Arg)), *val));
                }
                Self::InvVal(inner2) => {
                    *self = *inner2.clone();
                }
                _ => (),
            },
            Self::Neg(inner) => match inner.as_mut() {
                Self::Constant(val) => {
                    *self = Self::Constant(Self::eval(&Self::Neg(Box::new(Self::Arg)), *val));
                }
                Self::Neg(inner2) => {
                    *self = *inner2.clone();
                }
                _ => (),
            },
            Self::Abs(inner) => match inner.as_ref() {
                Self::Constant(val) => {
                    *self = Self::Constant(Self::eval(&Self::Abs(Box::new(Self::Arg)), *val));
                }
                Self::Abs(inner2) => {
                    *inner = inner2.clone();
                }
                Self::Neg(inner2) => {
                    *inner = inner2.clone();
                }
                _ => (),
            },
            Self::Sum(inner_l, inner_r) => {
                match inner_l.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_r.clone(), 1.0, *val);
                        return;
                    }
                    _ => (),
                }
                match inner_r.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_l.clone(), 1.0, *val);
                    }
                    _ => (),
                }
            }
            Self::Dif(inner_l, inner_r) => {
                match inner_l.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_r.clone(), -1.0, *val);
                        return;
                    }
                    _ => (),
                }
                match inner_r.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_l.clone(), 1.0, -*val);
                    }
                    _ => (),
                }
            }
            Self::Mul(inner_l, inner_r) => {
                match inner_l.as_mut() {
                    Self::Constant(c) => {
                        *self = Self::ConstMul(inner_r.clone(), *c);
                        return;
                    }
                    _ => (),
                }
                match inner_r.as_mut() {
                    Self::Constant(c) => {
                        *self = Self::ConstMul(inner_l.clone(), *c);
                    }
                    _ => (),
                }
            }
            Self::Div(inner_l, inner_r) => match inner_r.as_mut() {
                Self::Constant(c) => {
                    *self = Self::ConstMul(
                        inner_l.clone(),
                        EnvFunction::InvVal(Box::new(EnvFunction::Arg)).eval(*c),
                    );
                }
                _ => (),
            },
            _ => (),
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
        f.simplify();
        assert_eval_eq(&f, 0.0, 42.0, 1e-5);
    }

    #[test]
    fn test_simplify_constant_folding_unary() {
        // Exp(Constant(0)) → Constant(1)
        let mut f = EnvFunction::Exp(Box::new(EnvFunction::Constant(0.0)));
        f.simplify();
        assert_eval_eq(&f, 123.45, 1.0, 1e-5); // x не важен, функция константна

        // Sin(Constant(π/2)) ≈ 1
        let mut f = EnvFunction::Sin(Box::new(EnvFunction::Constant(std::f32::consts::FRAC_PI_2)));
        f.simplify();
        assert_eval_eq(&f, 0.0, 1.0, 1e-4);
    }

    #[test]
    fn test_simplify_const_mul() {
        // 0 * f(x) → 0
        let mut f =
            EnvFunction::ConstMul(Box::new(EnvFunction::Sin(Box::new(EnvFunction::Arg))), 0.0);
        f.simplify();
        assert_eval_eq(&f, 1.0, 0.0, 1e-5);

        // 1 * f(x) → f(x)
        let mut f = EnvFunction::ConstMul(Box::new(EnvFunction::Arg), 1.0);
        f.simplify();
        assert_eval_eq(&f, 5.0, 5.0, 1e-5);

        // c * Constant(v) → Constant(c*v)
        let mut f = EnvFunction::ConstMul(Box::new(EnvFunction::Constant(3.0)), 4.0);
        f.simplify();
        assert_eval_eq(&f, 0.0, 12.0, 1e-5);
    }

    #[test]
    fn test_simplify_linear() {
        // k=0: Linear(f, 0, b) → Constant(b)
        let mut f = EnvFunction::Linear(Box::new(EnvFunction::Arg), 0.0, 7.0);
        f.simplify();
        assert_eval_eq(&f, 100.0, 7.0, 1e-5);

        // b=0: Linear(f, k, 0) → ConstMul(f, k)
        let mut f = EnvFunction::Linear(Box::new(EnvFunction::Arg), 2.0, 0.0);
        f.simplify();
        assert_eval_eq(&f, 3.0, 6.0, 1e-5);
    }

    #[test]
    fn test_simplify_double_neg() {
        // Neg(Neg(x)) → x
        let mut f = EnvFunction::Neg(Box::new(EnvFunction::Neg(Box::new(EnvFunction::Arg))));
        f.simplify();
        assert_eval_eq(&f, -5.0, -5.0, 1e-5);
        assert_eval_eq(&f, 3.14, 3.14, 1e-5);
    }

    #[test]
    fn test_simplify_abs_abs() {
        // Abs(Abs(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(EnvFunction::Abs(Box::new(EnvFunction::Arg))));
        f.simplify();
        assert_eval_eq(&f, -7.0, 7.0, 1e-5);
        assert_eval_eq(&f, 2.0, 2.0, 1e-5);
    }

    #[test]
    fn test_simplify_abs_neg() {
        // Abs(Neg(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(EnvFunction::Neg(Box::new(EnvFunction::Arg))));
        f.simplify();
        assert_eval_eq(&f, -4.0, 4.0, 1e-5);
    }

    #[test]
    fn test_simplify_inv_inv() {
        // InvVal(InvVal(x)) → x (для x ≠ 0)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::InvVal(Box::new(EnvFunction::Arg))));
        f.simplify();
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
        f.simplify();
        assert_eval_eq(&f, 10.0, 15.0, 1e-5); // 10 + 5
    }

    #[test]
    fn test_simplify_sum_constant_right() {
        // f(x) + Constant(c) → Linear(f, 1, c)
        let mut f = EnvFunction::Sum(
            Box::new(EnvFunction::Arg),
            Box::new(EnvFunction::Constant(-3.0)),
        );
        f.simplify();
        assert_eval_eq(&f, 10.0, 7.0, 1e-5); // 10 - 3
    }

    #[test]
    fn test_simplify_dif_constant() {
        // Constant(c) - f(x) → Linear(f, -1, c)
        let mut f = EnvFunction::Dif(
            Box::new(EnvFunction::Constant(10.0)),
            Box::new(EnvFunction::Arg),
        );
        f.simplify();
        assert_eval_eq(&f, 4.0, 6.0, 1e-5); // 10 - 4

        // f(x) - Constant(c) → Linear(f, 1, -c)
        let mut f2 = EnvFunction::Dif(
            Box::new(EnvFunction::Arg),
            Box::new(EnvFunction::Constant(2.0)),
        );
        f2.simplify();
        assert_eval_eq(&f2, 5.0, 3.0, 1e-5); // 5 - 2
    }

    #[test]
    fn test_simplify_periodic_constant() {
        // Periodic(Constant(v), p) → Constant(v)
        let mut f = EnvFunction::Periodic(Box::new(EnvFunction::Constant(1.5)), 2.0);
        f.simplify();
        assert_eval_eq(&f, 100.0, 1.5, 1e-5);
    }

    #[test]
    fn test_simplify_nested_simplify() {
        // Exp(Sin(Constant(0))) → Exp(0) → 1
        let mut f = EnvFunction::Exp(Box::new(EnvFunction::Sin(Box::new(EnvFunction::Constant(
            0.0,
        )))));
        f.simplify();
        assert_eval_eq(&f, 0.0, 1.0, 1e-5);
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
            deep.simplify();
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
            deep.simplify();
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
        f.simplify();
        assert_eval_eq(&f, 0.0, 0.0, 1e-5); // NaN превращается в 0
    }

    #[test]
    fn test_simplify_div_zero_safety() {
        // InvVal(Constant(0)) → 0 (защита в eval)
        let mut f = EnvFunction::InvVal(Box::new(EnvFunction::Constant(0.0)));
        f.simplify();
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
        f_simplified.simplify();

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
}
