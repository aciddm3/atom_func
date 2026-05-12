mod functions;

pub const EPS: f32 = 1e-7;

#[derive(Debug, Clone, Copy, Default)]
pub struct EnvFunctionArguments {
    pub t: f32,
    pub prev_val: f32,
    pub freq: f32,
    pub vel: f32,
    pub p1: f32,
    pub p2: f32,
    pub p3: f32,
    pub p4: f32,
}

#[derive(Debug, Default, Clone)]
pub enum EnvFunction {
    // technical
    #[default]
    Arg,

    PrevVal,
    Frequency,
    Velocity,

    P1,
    P2,
    P3,
    P4,

    Constant(f32),
    // functions
    Id(Box<Self>),
    ConstMul(Box<Self>, f32),
    Exp(Box<Self>),
    Sigmoid(Box<Self>),
    Ln(Box<Self>),
    Sin(Box<Self>),
    Cos(Box<Self>),
    // transformation
    Linear(Box<Self>, f32, f32),
    Powf(Box<Self>, f32),
    Powi(Box<Self>, i32),
    Periodic(Box<Self>, f32),
    InvVal(Box<Self>),
    Neg(Box<Self>),
    Abs(Box<Self>),
    
    Oct(Box<Self>, Box<Self>),
    Sum(Box<Self>, Box<Self>),
    Dif(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    //by points
    // TODO CompiledLUT(Box<[f32; 256]>),
}

impl EnvFunction {
    #[inline]
    pub fn eval(&self, arg: EnvFunctionArguments) -> f32 {
        let res = match self {
            Self::Arg => arg.t,
            Self::Frequency => arg.freq,
            Self::PrevVal => arg.prev_val,
            Self::Velocity => arg.vel,
            Self::P1 => arg.p1,
            Self::P2 => arg.p2,
            Self::P3 => arg.p3,
            Self::P4 => arg.p4,
            Self::Constant(t) => *t,
            Self::Id(inner) => inner.eval(arg),
            Self::Exp(inner) => inner.eval(arg).exp(),
            Self::Sigmoid(inner) => functions::sigmoid(inner.eval(arg)),
            Self::Ln(inner) => inner.eval(arg).ln(),
            Self::Sin(inner) => functions::approx_sine(inner.eval(arg)),
            Self::Cos(inner) => functions::approx_cosine(inner.eval(arg)),
            Self::Linear(inner, k, b) => *k * inner.eval(arg) + *b,
            Self::Powf(inner, p) => inner.eval(arg).powf(*p),
            Self::Powi(inner, p) => inner.eval(arg).powi(*p),
            Self::Periodic(inner, p) => {
                let period = p.abs().max(1e-5);
                inner.eval(arg).rem_euclid(period)
            }
            Self::InvVal(inner) => {
                let inv = inner.eval(arg);
                if inv.abs() < EPS { 0.0 } else { 1.0 / inv }
            }
            Self::Neg(inner) => -inner.eval(arg),
            Self::Abs(inner) => inner.eval(arg).abs(),
            Self::Oct(inner, inner2) => 2_f32.powf(inner.eval(arg)) * inner2.eval(arg),
            Self::Sum(inner, inner2) => inner.eval(arg) + inner2.eval(arg),
            Self::Dif(inner, inner2) => inner.eval(arg) - inner2.eval(arg),
            Self::Mul(inner, inner2) => inner.eval(arg) * inner2.eval(arg),
            Self::Div(inner, inner2) => {
                let inv = inner2.eval(arg);
                if inv.abs() < EPS {
                    0.0
                } else {
                    inner.eval(arg) / inv
                }
            }
            Self::ConstMul(inner, c) => c * inner.eval(arg),
        };
        if res.is_finite() && !res.is_nan() {
            res
        } else {
            0.0
        }
    }
}

