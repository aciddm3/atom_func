mod functions;

pub const EPS: f32 = 1e-7;

#[derive(Debug, Default, Clone)]
pub enum EnvFunction {
    // technical
    #[default]
    Arg,
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

    Sum(Box<Self>, Box<Self>),
    Dif(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    //by points
    // TODO CompiledLUT(Box<[f32; 256]>),
}

impl EnvFunction {
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        let res = match self {
            Self::Arg => x,
            Self::Constant(t) => *t,
            Self::Id(inner) => inner.eval(x),
            Self::Exp(inner) => inner.eval(x).exp(),
            Self::Sigmoid(inner) => functions::sigmoid(inner.eval(x)),
            Self::Ln(inner) => inner.eval(x).ln(),
            Self::Sin(inner) => functions::approx_sine(inner.eval(x)),
            Self::Cos(inner) => functions::approx_cosine(inner.eval(x)),
            Self::Linear(inner, k, b) => *k * inner.eval(x) + *b,
            Self::Powf(inner, p) => inner.eval(x).powf(*p),
            Self::Powi(inner, p) => inner.eval(x).powi(*p),
            Self::Periodic(inner, p) => {
                let period = p.abs().max(1e-5);
                inner.eval(x.rem_euclid(period))
            }
            Self::InvVal(inner) => {
                let inv = inner.eval(x);
                if inv.abs() < EPS { 0.0 } else { 1.0 / inv }
            }
            Self::Neg(inner) => -inner.eval(x),
            Self::Abs(inner) => inner.eval(x).abs(),
            Self::Sum(inner, inner2) => inner.eval(x) + inner2.eval(x),
            Self::Dif(inner, inner2) => inner.eval(x) - inner2.eval(x),
            Self::Mul(inner, inner2) => inner.eval(x) * inner2.eval(x),
            Self::Div(inner, inner2) => {
                let inv = inner2.eval(x);
                if inv.abs() < EPS {
                    0.0
                } else {
                    inner.eval(x) / inv
                }
            }
            Self::ConstMul(inner, c) => c * inner.eval(x),
        };
        if res.is_finite() && !res.is_nan() {
            res
        } else {
            0.0
        }
    }
}
