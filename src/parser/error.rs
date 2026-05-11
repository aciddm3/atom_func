// parser/error.rs
use std::fmt::{self};

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub kind: ErrorKind,
    pub line: usize,
    pub column: usize,
    pub snippet: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorKind {
    UnexpectedChar(char),
    UnexpectedEof,
    InvalidNumber(String),
    UnknownSymbol(String),
    WrongArity {
        symbol: String,
        expected: usize,
        got: usize,
    },
    MismatchedParens,
    ConversionError(String),
}

impl std::error::Error for ParseError {}

pub type ParseResult<T> = Result<T, ParseError>;

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error_desc = match &self.kind {
            ErrorKind::UnexpectedChar(c) => format!("unexpected character '{}'", c),
            ErrorKind::UnexpectedEof => "unexpected end of file".to_string(),
            ErrorKind::InvalidNumber(s) => format!("invalid number \"{}\"", s),
            ErrorKind::UnknownSymbol(s) => format!("unknown symbol \"{}\"", s),
            ErrorKind::WrongArity {
                symbol,
                expected,
                got,
            } => {
                format!(
                    "\"{}\" expected {} argument(s), got {}",
                    symbol, expected, got
                )
            }
            ErrorKind::MismatchedParens => "mismatched parentheses".to_string(),
            ErrorKind::ConversionError(e) => format!("conversion error: {}", e),
        };

        write!(f, "error: {}|{}: {}", self.line, self.column, error_desc)
    }
}

#[cfg(test)]
mod test {
    use crate::parser::parse_func;

    #[test]
    fn check_error() {
        let a = parse_func("(* (+ arg .-) 0.3)").unwrap_err();
        println!("{a}");
    }
}
