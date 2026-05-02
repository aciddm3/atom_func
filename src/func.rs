mod functions;

const EPS: f32 = 1e-7;

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
            Self::Id(inner) => Self::eval(inner, x),
            Self::Exp(inner) => Self::eval(inner, x).exp(),
            Self::Sigmoid(inner) => functions::sigmoid(Self::eval(inner, x)),
            Self::Ln(inner) => Self::eval(inner, x).ln(),
            Self::Sin(inner) => Self::eval(inner, x).sin(),
            Self::Cos(inner) => Self::eval(inner, x).cos(),
            Self::Linear(inner, k, b) => *k * Self::eval(inner, x) + *b,
            Self::Powf(inner, p) => Self::eval(inner, x).powf(*p),
            Self::Powi(inner, p) => Self::eval(inner, x).powi(*p),
            Self::Periodic(inner, p) => {
                let period = p.abs().max(1e-5);
                Self::eval(inner, x.rem_euclid(period))
            }
            Self::InvVal(inner) => {
                let inv = Self::eval(inner, x);
                if inv.abs() < EPS { 0.0 } else { 1.0 / inv }
            }
            Self::Neg(inner) => -Self::eval(inner, x),
            Self::Abs(inner) => Self::eval(inner, x).abs(),
            Self::Sum(inner, inner2) => Self::eval(inner, x) + Self::eval(inner2, x),
            Self::Dif(inner, inner2) => Self::eval(inner, x) - Self::eval(inner2, x),
            Self::Mul(inner, inner2) => Self::eval(inner, x) * Self::eval(inner2, x),
            Self::Div(inner, inner2) => {
                let inv = Self::eval(inner2, x);
                if inv.abs() < EPS {
                    0.0
                } else {
                    Self::eval(inner, x) / inv
                }
            }
            Self::ConstMul(inner, c) => c * Self::eval(inner, x),
        };
        if res.is_finite() && !res.is_nan() {
            res
        } else {
            0.0
        }
    }
}
