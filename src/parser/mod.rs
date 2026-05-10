// parser/mod.rs
pub mod ast;
pub mod convert;
pub mod error;
pub mod lexer;

use crate::func::EnvFunction;
use error::ParseResult;

pub fn parse_func(input: &str) -> ParseResult<EnvFunction> {
    let mut lexer = lexer::Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let line = lexer.line;
    let col = lexer.column;

    let mut parser = ast::Parser::from_tokens(tokens, line, col);
    let sexpr = parser.parse()?;

    convert::to_env(sexpr)
}

/// Парсит несколько выражений (разделённых новой строкой или пробелами)
pub fn parse_env_batch(input: &str) -> ParseResult<Vec<EnvFunction>> {
    // Можно расширить при необходимости
    Ok(vec![parse_func(input)?])
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_nested_parse() {
        let input = "(sin (const-mul arg 2.0))";
        println!("Sexpr parsing test: parsing {input}");
        let result = parse_func(input);

        assert!(result.is_ok(), "Ошибка: {:?}", result);

        if let Ok(EnvFunction::Sin(inner)) = result {
            if let EnvFunction::ConstMul(inner2, factor) = *inner {
                assert!(matches!(*inner2, EnvFunction::Arg));
                assert!((factor - 2.0).abs() < 1e-5);
            } else {
                panic!("Ожидался ConstMul внутри Sin");
            }
        } else {
            panic!("Ожидался Sin на верхнем уровне");
        }
    }

    #[test]
    fn sub_test() {
        let input = "(sigmoid (- arg 2))";
        println!("Sexpr parsing test: parsing {input}");
        let result = parse_func(input);

        assert!(result.is_ok(), "Error: {:?}", result);

        if let Ok(EnvFunction::Sigmoid(inner)) = result {
            if let EnvFunction::Dif(inner2, inner3) = *inner {
                assert!(matches!(*inner2, EnvFunction::Arg));
                assert!(matches!(*inner3, EnvFunction::Constant(2.0)));
            } else {
                panic!("Expected Difference");
            }
        } else {
            panic!("Expected Sigmoid");
        }
    }

    #[test]
    fn numbers_test() {
        let input = [
            "-2", "-0", "1", "-3.14", "3.14", "-3e-8", "-3e8", "3e-8", "3e8",
        ];
        let expected = [-2.0, 0.0, 1.0, -3.14, 3.14, -3e-8, -3e8, 3e-8, 3e8];

        for (src, exp) in input.iter().zip(expected) {
            let wrapped_res = parse_func(src);
            if let Ok(res) = wrapped_res {
                match res {
                    EnvFunction::Constant(v) => assert!(
                        (v - exp).abs() < 1e-5,
                        "failed at '{}': got {}, expected {}",
                        src,
                        v,
                        exp
                    ),
                    other => panic!("failed at '{}': expected Constant, got {:?}", src, other),
                }
            } else {
                panic!("Error {:?}", wrapped_res);
            }
        }
    }

    #[cfg(test)]

    // ===== 🔹 ТЕХНИЧЕСКИЕ =====
    
    #[test]
    fn test_arg() {
        let f = parse_func("arg").unwrap();
        assert!(matches!(f, EnvFunction::Arg));
    }

    #[test]
    fn test_constant() {
        let cases = [
            ("0.0", 0.0),
            ("1.0", 1.0),
            ("-3.14", -3.14),
            ("1e-5", 1e-5),
            ("2.5E3", 2500.0),
        ];
        for (src, expected) in cases {
            let f = parse_func(src).unwrap();
            if let EnvFunction::Constant(v) = f {
                assert!((v - expected).abs() < 1e-5, "failed at '{}'", src);
            } else {
                panic!("expected Constant at '{}', got {:?}", src, f);
            }
        }
    }

    // ===== 🔸 УНАРНЫЕ ФУНКЦИИ =====
    
    #[test]
    fn test_id() {
        let f = parse_func("(id arg)").unwrap();
        assert!(matches!(f, EnvFunction::Id(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_exp() {
        let f = parse_func("(exp arg)").unwrap();
        assert!(matches!(f, EnvFunction::Exp(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_sigmoid() {
        let f = parse_func("(sigmoid arg)").unwrap();
        assert!(matches!(f, EnvFunction::Sigmoid(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_ln() {
        let f = parse_func("(ln arg)").unwrap();
        assert!(matches!(f, EnvFunction::Ln(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_sin() {
        let f = parse_func("(sin arg)").unwrap();
        assert!(matches!(f, EnvFunction::Sin(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_cos() {
        let f = parse_func("(cos arg)").unwrap();
        assert!(matches!(f, EnvFunction::Cos(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_neg() {
        let f = parse_func("(neg arg)").unwrap();
        assert!(matches!(f, EnvFunction::Neg(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_abs() {
        let f = parse_func("(abs arg)").unwrap();
        assert!(matches!(f, EnvFunction::Abs(inner) if matches!(*inner, EnvFunction::Arg)));
    }

    #[test]
    fn test_inv_val() {
        // Проверяем оба алиаса
        let f1 = parse_func("(inv arg)").unwrap();
        let f2 = parse_func("(inv-val arg)").unwrap();
        assert!(matches!(f1, EnvFunction::InvVal(_)));
        assert!(matches!(f2, EnvFunction::InvVal(_)));
    }

    // ===== 🔹 БИНАРНЫЕ ОПЕРАЦИИ =====
    
    #[test]
    fn test_sum() {
        let f = parse_func("(+ arg 1.0)").unwrap();
        assert!(matches!(f, EnvFunction::Sum(a, b) 
            if matches!(*a, EnvFunction::Arg) && matches!(*b, EnvFunction::Constant(_))));
    }

    #[test]
    fn test_dif() {
        let f = parse_func("(- arg 2.0)").unwrap();
        assert!(matches!(f, EnvFunction::Dif(a, b) 
            if matches!(*a, EnvFunction::Arg) && matches!(*b, EnvFunction::Constant(_))));
    }

    #[test]
    fn test_mul() {
        let f = parse_func("(* arg 0.5)").unwrap();
        assert!(matches!(f, EnvFunction::Mul(a, b) 
            if matches!(*a, EnvFunction::Arg) && matches!(*b, EnvFunction::Constant(_))));
    }

    #[test]
    fn test_div() {
        let f = parse_func("(/ arg 2.0)").unwrap();
        assert!(matches!(f, EnvFunction::Div(a, b) 
            if matches!(*a, EnvFunction::Arg) && matches!(*b, EnvFunction::Constant(_))));
    }

    // ===== 🔸 ПАРАМЕТРИЧЕСКИЕ УНАРНЫЕ =====
    
    #[test]
    fn test_const_mul() {
        let cases = [
            ("(const-mul arg 1.0)", 1.0),
            ("(const-mul arg -2.5)", -2.5),
            ("(const-mul arg 1e-3)", 1e-3),
        ];
        for (src, expected_k) in cases {
            let f = parse_func(src).unwrap();
            if let EnvFunction::ConstMul(inner, k) = f {
                assert!(matches!(*inner, EnvFunction::Arg), "inner should be Arg at '{}'", src);
                assert!((k - expected_k).abs() < 1e-5, "k mismatch at '{}'", src);
            } else {
                panic!("expected ConstMul at '{}', got {:?}", src, f);
            }
        }
    }

    #[test]
    fn test_linear() {
        let f = parse_func("(linear arg 2.0 -0.5)").unwrap();
        if let EnvFunction::Linear(inner, scale, offset) = f {
            assert!(matches!(*inner, EnvFunction::Arg));
            assert!((scale - 2.0).abs() < 1e-5);
            assert!((offset + 0.5).abs() < 1e-5);
        } else {
            panic!("expected Linear, got {:?}", f);
        }
    }

    #[test]
    fn test_powf() {
        let f = parse_func("(powf arg 0.5)").unwrap();
        if let EnvFunction::Powf(inner, exp) = f {
            assert!(matches!(*inner, EnvFunction::Arg));
            assert!((exp - 0.5).abs() < 1e-5);
        } else {
            panic!("expected Powf, got {:?}", f);
        }
    }

    #[test]
    fn test_powi() {
        let cases = [
            ("(powi arg 2)", 2),
            ("(powi arg -3)", -3),
            ("(powi arg 0)", 0),
        ];
        for (src, expected_exp) in cases {
            let f = parse_func(src).unwrap();
            if let EnvFunction::Powi(inner, exp) = f {
                assert!(matches!(*inner, EnvFunction::Arg), "inner should be Arg at '{}'", src);
                assert_eq!(exp, expected_exp, "exp mismatch at '{}'", src);
            } else {
                panic!("expected Powi at '{}', got {:?}", src, f);
            }
        }
    }

    #[test]
    fn test_periodic() {
        let f = parse_func("(periodic arg 6.283185)").unwrap();
        if let EnvFunction::Periodic(inner, period) = f {
            assert!(matches!(*inner, EnvFunction::Arg));
            assert!((period - 6.283185).abs() < 1e-5);
        } else {
            panic!("expected Periodic, got {:?}", f);
        }
    }

    // ===== 🔹 КОМПОЗИЦИИ И ГРАНИЧНЫЕ СЛУЧАИ =====
    

    #[test]
    fn test_nested_binary() {
        let f = parse_func("(+ (* arg 2.0) (- arg 1.0))").unwrap();
        assert!(matches!(f, EnvFunction::Sum(_, _)));
    }

    #[test]
    fn test_mixed_composition() {
        let f = parse_func("(linear (sin (const-mul arg 440.0)) 0.5 0.0)").unwrap();
        if let EnvFunction::Linear(inner, s, o) = f {
            assert!(matches!(*inner, EnvFunction::Sin(_)));
            assert!((s - 0.5).abs() < 1e-5);
            assert!((o - 0.0).abs() < 1e-5);
        } else {
            panic!("expected Linear, got {:?}", f);
        }
    }

    #[test]
    fn test_minus_operator_vs_negative_number() {
        // Оператор "-"
        let f1 = parse_func("(- arg 1.0)").unwrap();
        assert!(matches!(f1, EnvFunction::Dif(_, _)));
        
        // Отрицательное число
        let f2 = parse_func("(const-mul arg -2.5)").unwrap();
        if let EnvFunction::ConstMul(_, k) = f2 {
            assert!((k + 2.5).abs() < 1e-5);
        } else {
            panic!("expected ConstMul with negative number");
        }
    }

    #[test]
    fn test_math_constants() {
        use std::f32::consts::{PI, E, TAU};
    
        let cases = [
            ("PI", PI),      // case-insensitive
            ("π", PI),       // юникод!
            ("E", E),
            ("TAU", TAU),
            ("τ", TAU),
            ("PHI", 1.618033988749895),
        ];
    
        for (src, expected) in cases {
            let f = parse_func(src).unwrap();
            if let EnvFunction::Constant(v) = f {
                assert!((v - expected).abs() < 1e-5, "failed at '{}'", src);
            } else {
                panic!("expected Constant at '{}', got {:?}", src, f);
            }
        }
    }
}
