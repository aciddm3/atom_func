pub mod func;

#[cfg(test)]
mod tests {
    use crate::func::EnvFunction;

    #[test]
    fn test_arg() {
        let func = EnvFunction::Arg;
        let vc = (-32000..32000).map(|s| s as f32).collect::<Vec<_>>();
        assert_eq!(vc, vc.iter().map(|&x| func.eval(x)).collect::<Vec<_>>());
    }

    #[test]
    fn test_id() {
        let func = EnvFunction::Id(Box::new(EnvFunction::Arg));
        let vc = (-32000..32000).map(|s| s as f32).collect::<Vec<_>>();
        assert_eq!(vc, vc.iter().map(|&x| func.eval(x)).collect::<Vec<_>>());
    }

    #[test]
    fn test_sine() {
        let func = EnvFunction::Sin(Box::new(EnvFunction::Arg));
        let vc = (-32000..32000).map(|s| s as f32).collect::<Vec<_>>();
        assert_eq!(
            vc.iter().map(|&x| x.sin()).collect::<Vec<_>>(),
            vc.iter().map(|&x| func.eval(x)).collect::<Vec<_>>()
        );
    }

    // 🔧 Хелпер: проверка, что функция после simplify даёт те же результаты
    fn assert_eval_eq(f: &EnvFunction, x: f32, expected: f32, tolerance: f32) {
        let actual = f.eval(x);
        assert!(
            (actual - expected).abs() < tolerance,
            "f({}) = {}, expected {}",
            x, actual, expected
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
        let mut f = EnvFunction::ConstMul(Box::new(EnvFunction::Sin(Box::new(EnvFunction::Arg))), 0.0);
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

    // ─────────────────────────────────────────────────────
    // 🔁 Двойные инверсии
    // ─────────────────────────────────────────────────────

    #[test]
    fn test_simplify_double_neg() {
        // Neg(Neg(x)) → x
        let mut f = EnvFunction::Neg(Box::new(
            EnvFunction::Neg(Box::new(EnvFunction::Arg))
        ));
        f.simplify();
        assert_eval_eq(&f, -5.0, -5.0, 1e-5);
        assert_eval_eq(&f, 3.14, 3.14, 1e-5);
    }

    #[test]
    fn test_simplify_abs_abs() {
        // Abs(Abs(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(
            EnvFunction::Abs(Box::new(EnvFunction::Arg))
        ));
        f.simplify();
        assert_eval_eq(&f, -7.0, 7.0, 1e-5);
        assert_eval_eq(&f, 2.0, 2.0, 1e-5);
    }

    #[test]
    fn test_simplify_abs_neg() {
        // Abs(Neg(x)) → Abs(x)
        let mut f = EnvFunction::Abs(Box::new(
            EnvFunction::Neg(Box::new(EnvFunction::Arg))
        ));
        f.simplify();
        assert_eval_eq(&f, -4.0, 4.0, 1e-5);
    }

    #[test]
    fn test_simplify_inv_inv() {
        // InvVal(InvVal(x)) → x (для x ≠ 0)
        let mut f = EnvFunction::InvVal(Box::new(
            EnvFunction::InvVal(Box::new(EnvFunction::Arg))
        ));
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
        let mut f = EnvFunction::Periodic(
            Box::new(EnvFunction::Constant(1.5)),
            2.0,
        );
        f.simplify();
        assert_eval_eq(&f, 100.0, 1.5, 1e-5);
    }

    #[test]
    fn test_simplify_nested_simplify() {
        // Exp(Sin(Constant(0))) → Exp(0) → 1
        let mut f = EnvFunction::Exp(Box::new(
            EnvFunction::Sin(Box::new(EnvFunction::Constant(0.0)))
        ));
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
                x, original, simplified
            );
        }
    }
}
