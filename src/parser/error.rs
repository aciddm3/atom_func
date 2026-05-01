// parser/error.rs
use std::fmt;

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

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Ошибка парсинга [{}:{}]: {:?}\n\t -> {}",
            self.line, self.column, self.kind, self.snippet
        )
    }
}

impl std::error::Error for ParseError {}

pub type ParseResult<T> = Result<T, ParseError>;
