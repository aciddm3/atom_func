// parser/lexer.rs
use crate::parser::error::{ErrorKind, ParseError, ParseResult};

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    LParen,
    RParen,
    Symbol(String),
    Number(f32),
}

pub struct Lexer<'a> {
    input: &'a str,
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    pub line: usize,
    pub column: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            input,
            chars: input.chars().peekable(),
            line: 1,
            column: 1,
        }
    }

    pub fn tokenize(&mut self) -> ParseResult<Vec<Token>> {
        let mut tokens = Vec::with_capacity(64); // pre-allocate

        // parser/lexer.rs -> в методе tokenize()
        while let Some(&ch) = self.chars.peek() {
            match ch {
                ' ' | '\t' | '\r' => {
                    self.advance();
                }
                '\n' => {
                    self.advance_line();
                }
                ';' => {
                    self.skip_comment();
                }
                '(' => {
                    tokens.push(Token::LParen);
                    self.advance();
                }
                ')' => {
                    tokens.push(Token::RParen);
                    self.advance();
                }

                // 🔑 Явная обработка минуса с ПРАВИЛЬНЫМ lookahead
                '-' => {
                    // Клонируем итератор, чтобы заглянуть вперёд, не потребляя символы в основном
                    let mut lookahead = self.chars.clone();
                    lookahead.next(); // пропускаем '-' в клоне
                    if let Some(&next) = lookahead.peek() {
                        if next.is_ascii_digit() || next == '.' {
                            // Это отрицательное число: -2, -3.14, -.5, -1e-10
                            tokens.push(self.parse_number()?);
                        } else {
                            // Это оператор или символ: -, -foo, -=
                            tokens.push(self.parse_symbol()?);
                        }
                    } else {
                        // Конец ввода сразу после '-'
                        tokens.push(Token::Symbol("-".into()));
                        self.advance();
                    }
                }

                // Цифры и точка (без минуса!)
                c if c.is_ascii_digit() || c == '.' => {
                    tokens.push(self.parse_number()?);
                }

                // Всё остальное, что может быть символом
                c if is_symbol_start(c) => {
                    tokens.push(self.parse_symbol()?);
                }

                _ => {
                    return Err(ParseError {
                        kind: ErrorKind::UnexpectedChar(ch),
                        line: self.line,
                        column: self.column,
                        snippet: self.snippet(),
                    });
                }
            }
        }
        Ok(tokens)
    }

    fn parse_number(&mut self) -> ParseResult<Token> {
        let start_pos = (self.line, self.column);
        let mut num = String::with_capacity(16);

        if self.peek() == Some('-') {
            num.push(self.advance().unwrap());
        }

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                num.push(self.advance().unwrap());
            } else {
                break;
            }
        }

        if self.peek() == Some('.') {
            num.push(self.advance().unwrap());
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    num.push(self.advance().unwrap());
                } else {
                    break;
                }
            }
        }

        // Экспонента (опционально)
        if let Some('e' | 'E') = self.peek() {
            num.push(self.advance().unwrap());
            if let Some('+' | '-') = self.peek() {
                num.push(self.advance().unwrap());
            }
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    num.push(self.advance().unwrap());
                } else {
                    break;
                }
            }
        }

        num.parse::<f32>()
            .map(Token::Number)
            .map_err(|_| ParseError {
                kind: ErrorKind::InvalidNumber(num),
                line: start_pos.0,
                column: start_pos.1,
                snippet: self.snippet(),
            })
    }

    fn parse_symbol(&mut self) -> ParseResult<Token> {
        let mut sym = String::with_capacity(32);

        while let Some(c) = self.peek() {
            if is_symbol_char(c) {
                sym.push(self.advance().unwrap());
            } else {
                break;
            }
        }

        Ok(Token::Symbol(sym))
    }

    fn skip_comment(&mut self) {
        while let Some(c) = self.peek() {
            if c == '\n' {
                break;
            }
            self.advance();
        }
    }

    #[inline]
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().copied()
    }

    #[inline]
    fn advance(&mut self) -> Option<char> {
        let c = self.chars.next()?;
        self.column += 1;
        Some(c)
    }

    #[inline]
    fn advance_line(&mut self) {
        self.chars.next();
        self.line += 1;
        self.column = 1;
    }

    fn snippet(&self) -> String {
        self.input
            .lines()
            .nth(self.line - 1)
            .unwrap_or("")
            .chars()
            .take(self.column + 10)
            .collect()
    }
}

#[inline]
fn is_symbol_start(c: char) -> bool {
    c.is_alphabetic()
        || c == '_'
        || c == '+'
        || c == '*'
        || c == '/'
        || c == '='
        || c == '!'
        || c == '<'
        || c == '>'
        || c == '?'
}

#[inline]
fn is_symbol_char(c: char) -> bool {
    is_symbol_start(c) || c.is_ascii_digit() || c == '-' || c == '.'
}
