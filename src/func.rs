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
    #[inline]
    fn simplify_with_depth(&mut self, depth: usize) {
        if depth > 32 {
            return;
        } // Stack overflow protection
        match self {
            Self::Id(inner) => {
                *self = *inner.clone();
                self.simplify_with_depth(depth + 1);
            }
            Self::ConstMul(inner, c) => {
                inner.simplify_with_depth(depth + 1);
                if *c == 0.0 {
                    *self = Self::Constant(0.0);
                    return;
                }
                if *c == 1.0 {
                    *self = *inner.clone();
                    return;
                }
                match inner.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Constant(Self::eval(
                            &Self::ConstMul(Box::new(Self::Arg), *c),
                            *val,
                        ));
                    }
                    Self::ConstMul(inner2, b) => *self = Self::ConstMul(inner2.clone(), *b * *c),
                    _ => (),
                }
            }
            Self::Exp(inner) => {
                inner.simplify_with_depth(depth + 1);
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Exp(Box::new(Self::Arg)), val));
                }
            }
            Self::Sigmoid(inner) => {
                inner.simplify_with_depth(depth + 1);
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Sigmoid(Box::new(Self::Arg)), val));
                }
            }
            Self::Ln(inner) => {
                inner.simplify_with_depth(depth + 1);
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Ln(Box::new(Self::Arg)), val));
                }
            }
            Self::Sin(inner) => {
                inner.simplify_with_depth(depth + 1);
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Sin(Box::new(Self::Arg)), val));
                }
            }
            Self::Cos(inner) => {
                inner.simplify_with_depth(depth + 1);
                match inner.as_ref() {
                    Self::Constant(val) => {
                        *self = Self::Constant(Self::eval(&Self::Cos(Box::new(Self::Arg)), *val));
                    }
                    Self::Abs(inner2) => {
                        *inner = inner2.clone();
                        self.simplify_with_depth(depth + 1);
                    }
                    Self::Neg(inner2) => {
                        *inner = inner2.clone();
                        self.simplify_with_depth(depth + 1)
                    }
                    _ => (),
                }
            }
            Self::Linear(inner, k, b) => {
                if *k == 0.0 {
                    *self = Self::Constant(*b);
                    return;
                }
                if *b == 0.0 {
                    *self = Self::ConstMul(inner.clone(), *k);
                    return;
                }
                inner.simplify_with_depth(depth + 1);
                if let Self::Constant(val) = **inner {
                    *self =
                        Self::Constant(Self::eval(&Self::Linear(Box::new(Self::Arg), *k, *b), val));
                }
            }
            Self::Powf(inner, p) => {
                inner.simplify_with_depth(depth + 1);
                if let Self::Constant(val) = **inner {
                    *self = Self::Constant(Self::eval(&Self::Powf(Box::new(Self::Arg), *p), val));
                }
            }
            Self::Powi(inner, p) => {
                inner.simplify_with_depth(depth + 1);
                match inner.as_mut() {
                    Self::Constant(val) => {
                        *self =
                            Self::Constant(Self::eval(&Self::Powi(Box::new(Self::Arg), *p), *val));
                        return;
                    }
                    _ => (),
                }
                if *p % 2 == 0 {
                    match inner.as_mut() {
                        Self::Abs(inner) => {
                            *inner = inner.clone();
                            self.simplify_with_depth(depth + 1);
                        }
                        Self::Neg(inner) => {
                            *inner = inner.clone();
                            self.simplify_with_depth(depth + 1)
                        }
                        _ => (),
                    }
                }
            }
            Self::Periodic(inner, t) => {
                inner.simplify_with_depth(depth + 1);
                if let Self::Constant(val) = **inner {
                    *self =
                        Self::Constant(Self::eval(&Self::Periodic(Box::new(Self::Arg), *t), val));
                }
            }
            Self::InvVal(inner) => {
                inner.simplify_with_depth(depth + 1);
                match inner.as_mut() {
                    Self::Constant(val) => {
                        *self =
                            Self::Constant(Self::eval(&Self::InvVal(Box::new(Self::Arg)), *val));
                        self.simplify_with_depth(depth + 1);
                    }
                    Self::InvVal(inner2) => {
                        *self = *inner2.clone();
                        self.simplify_with_depth(depth + 1);
                    }
                    _ => (),
                }
            }
            Self::Neg(inner) => {
                inner.simplify_with_depth(depth + 1);
                match inner.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Constant(Self::eval(&Self::Neg(Box::new(Self::Arg)), *val));
                        self.simplify_with_depth(depth + 1);
                    }
                    Self::Neg(inner2) => {
                        *self = *inner2.clone();
                    }
                    _ => (),
                }
            }
            Self::Abs(inner) => {
                inner.simplify_with_depth(depth + 1);
                match inner.as_ref() {
                    Self::Constant(val) => {
                        *self = Self::Constant(Self::eval(&Self::Abs(Box::new(Self::Arg)), *val));
                    }
                    Self::Abs(inner2) => {
                        *inner = inner2.clone();
                        self.simplify_with_depth(depth + 1);
                    }
                    Self::Neg(inner2) => {
                        *inner = inner2.clone();
                        self.simplify_with_depth(depth + 1)
                    }
                    _ => (),
                }
            }
            Self::Sum(inner_l, inner_r) => {
                match inner_l.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_r.clone(), 1.0, *val);
                        self.simplify_with_depth(depth + 1);
                        return;
                    }
                    _ => (),
                }
                match inner_r.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_l.clone(), 1.0, *val);
                        self.simplify_with_depth(depth + 1);
                    }
                    _ => (),
                }
            }
            Self::Dif(inner_l, inner_r) => {
                match inner_l.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_r.clone(), -1.0, *val);
                        self.simplify_with_depth(depth + 1);
                        return;
                    }
                    _ => (),
                }
                match inner_r.as_mut() {
                    Self::Constant(val) => {
                        *self = Self::Linear(inner_l.clone(), 1.0, -*val);
                        self.simplify_with_depth(depth + 1);
                    }
                    _ => (),
                }
            }
            //Self::Mul(inner_l, inner_r) => (),
            //Self::Div(inner_l, inner_r) => (),
            _ => (),
        }
    }

    #[inline]
    pub fn simplify(&mut self) {
        self.simplify_with_depth(0);
    }
}
