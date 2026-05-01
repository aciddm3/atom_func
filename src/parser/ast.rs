// parser/ast.rs
use crate::parser::lexer::Token;
use crate::parser::error::{ParseError, ErrorKind, ParseResult};

#[derive(Debug, Clone, PartialEq)]
pub enum SExpr {
    Atom(Atom),
    List(Vec<SExpr>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Atom {
    Symbol(String),
    Number(f32),
}

pub struct Parser {
    tokens: std::vec::IntoIter<Token>,
    current: Option<Token>,
    line: usize,
    column: usize,
}

impl Parser {
    pub fn from_tokens(tokens: Vec<Token>, line: usize, col: usize) -> Self {
        let mut iter = tokens.into_iter();
        let current = iter.next();
        Self { tokens: iter, current, line, column: col }
    }

    pub fn parse(&mut self) -> ParseResult<SExpr> {
        self.skip_to_next();
        self.parse_expr()
    }

    fn parse_expr(&mut self) -> ParseResult<SExpr> {
        match self.current.take() {
            Some(Token::LParen) => {
                let mut list = Vec::with_capacity(8);
                self.advance();
                
                while !matches!(self.current, Some(Token::RParen) | None) {
                    list.push(self.parse_expr()?);
                    self.skip_to_next();
                }
                
                if matches!(self.current, Some(Token::RParen)) {
                    self.advance();
                    Ok(SExpr::List(list))
                } else {
                    Err(self.error(ErrorKind::MismatchedParens))
                }
            }
            Some(Token::Number(n)) => {
                self.advance();
                Ok(SExpr::Atom(Atom::Number(n)))
            }
            Some(Token::Symbol(s)) => {
                self.advance();
                Ok(SExpr::Atom(Atom::Symbol(s)))
            }
            None => Err(self.error(ErrorKind::UnexpectedEof)),
            _ => Err(self.error(ErrorKind::UnexpectedChar('?'))),
        }
    }

    #[inline]
    fn advance(&mut self) {
        self.current = self.tokens.next();
    }
    
    #[inline]
    fn skip_to_next(&mut self) {
        // Токены уже очищены лексером от пробелов/комментариев
    }
    
    fn error(&self, kind: ErrorKind) -> ParseError {
        ParseError {
            kind,
            line: self.line,
            column: self.column,
            snippet: String::new(),
        }
    }
}