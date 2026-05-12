// parser/convert.rs
use crate::func::EnvFunction;
use crate::parser::ast::{Atom, SExpr};
use crate::parser::error::{ErrorKind, ParseError, ParseResult};

pub fn to_env(expr: SExpr) -> ParseResult<EnvFunction> {
    match expr {
        SExpr::Atom(Atom::Number(n)) => Ok(EnvFunction::Constant(n)),
        SExpr::Atom(Atom::Symbol(s)) => symbol_to_env(&s),
        SExpr::List(list) => list_to_env(list),
    }
}

fn symbol_to_env(sym: &str) -> ParseResult<EnvFunction> {
    match sym {
        "arg" => Ok(EnvFunction::Arg),
        "prval" => Ok(EnvFunction::PrevVal),
        "vel" => Ok(EnvFunction::Velocity),
        "freq" => Ok(EnvFunction::Frequency),
        
        "p1" => Ok(EnvFunction::P1),
        "p2" => Ok(EnvFunction::P2),
        "p3" => Ok(EnvFunction::P3),
        "p4" => Ok(EnvFunction::P4),

        "PI" | "π" => Ok(EnvFunction::Constant(std::f32::consts::PI)),
        "TAU" | "TWOPI" | "τ" => Ok(EnvFunction::Constant(std::f32::consts::TAU)),
        "E" => Ok(EnvFunction::Constant(std::f32::consts::E)),
        "PHI" | "φ" => Ok(EnvFunction::Constant(1.618033988749895)),
        _ => Err(ParseError {
            kind: ErrorKind::UnknownSymbol(sym.to_string()),
            line: 0,
            column: 0,
            snippet: String::new(),
        }),
    }
}

fn list_to_env(list: Vec<SExpr>) -> ParseResult<EnvFunction> {
    if list.is_empty() {
        return Err(ParseError {
            kind: ErrorKind::UnexpectedEof,
            line: 0,
            column: 0,
            snippet: String::new(),
        });
    }

    let (func_name, args) = match &list[0] {
        SExpr::Atom(Atom::Symbol(s)) => (s.as_str(), &list[1..]),
        _ => {
            return Err(ParseError {
                kind: ErrorKind::UnknownSymbol("<non-symbol>".into()),
                line: 0,
                column: 0,
                snippet: String::new(),
            });
        }
    };

    macro_rules! unary {
        ($ctor:ident) => {{
            require_args(args, 1, func_name)?;
            Ok(EnvFunction::$ctor(Box::new(to_env(args[0].clone())?)))
        }};
    }

    macro_rules! binary {
        ($ctor:ident) => {{
            require_args(args, 2, func_name)?;
            Ok(EnvFunction::$ctor(
                Box::new(to_env(args[0].clone())?),
                Box::new(to_env(args[1].clone())?),
            ))
        }};
    }

    macro_rules! unary_param {
        ($ctor:ident, $idx:expr) => {{
            require_args(args, 2, func_name)?;
            let param = extract_f32(&args[$idx])?;
            Ok(EnvFunction::$ctor(
                Box::new(to_env(args[0].clone())?),
                param,
            ))
        }};
    }

    match func_name {
        // === Unary functions ===
        "id" => unary!(Id),
        "exp" => unary!(Exp),
        "sigmoid" => unary!(Sigmoid),
        "ln" => unary!(Ln),
        "sin" => unary!(Sin),
        "cos" => unary!(Cos),
        "neg" => unary!(Neg),
        "abs" => unary!(Abs),
        "inv" | "inv-val" => unary!(InvVal),

        // === Unary + f32 param ===
        "oct" => unary_param!(Oct, 1),
        "const-mul" => unary_param!(ConstMul, 1),
        "powf" => unary_param!(Powf, 1),
        "periodic" => unary_param!(Periodic, 1),

        // === Unary + f32, f32 params ===
        "linear" => {
            require_args(args, 3, func_name)?;
            let scale = extract_f32(&args[1])?;
            let offset = extract_f32(&args[2])?;
            Ok(EnvFunction::Linear(
                Box::new(to_env(args[0].clone())?),
                scale,
                offset,
            ))
        }

        // === Unary + i32 param ===
        "powi" => {
            require_args(args, 2, func_name)?;
            let exp = extract_i32(&args[1])?;
            Ok(EnvFunction::Powi(Box::new(to_env(args[0].clone())?), exp))
        }

        // === Binary functions ===
        "+" => binary!(Sum),
        "-" => binary!(Dif),
        "*" => binary!(Mul),
        "/" => binary!(Div),

        _ => Err(ParseError {
            kind: ErrorKind::UnknownSymbol(func_name.to_string()),
            line: 0,
            column: 0,
            snippet: String::new(),
        }),
    }
}

#[inline]
fn require_args(args: &[SExpr], expected: usize, func: &str) -> ParseResult<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(ParseError {
            kind: ErrorKind::WrongArity {
                symbol: func.to_string(),
                expected,
                got: args.len(),
            },
            line: 0,
            column: 0,
            snippet: String::new(),
        })
    }
}

#[inline]
fn extract_f32(expr: &SExpr) -> ParseResult<f32> {
    match expr {
        SExpr::Atom(Atom::Number(n)) => Ok(*n),
        _ => Err(ParseError {
            kind: ErrorKind::ConversionError("expected f32 literal".into()),
            line: 0,
            column: 0,
            snippet: String::new(),
        }),
    }
}

#[inline]
fn extract_i32(expr: &SExpr) -> ParseResult<i32> {
    match expr {
        SExpr::Atom(Atom::Number(n)) => Ok(*n as i32),
        _ => Err(ParseError {
            kind: ErrorKind::ConversionError("expected integer literal".into()),
            line: 0,
            column: 0,
            snippet: String::new(),
        }),
    }
}
