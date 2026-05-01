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
}
