use std::f32::consts::{FRAC_PI_2, PI, TAU};
const FRAC_3PI_2: f32 = PI + FRAC_PI_2;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 15.0 {
        1.0
    } else if x <= 15.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

#[inline]
pub fn approx_sine(mut x: f32) -> f32 {
    x = x.rem_euclid(TAU);

    let (x_reduced, sign) = if x <= PI {
        if x <= FRAC_PI_2 {
            (x, 1.0)
        } else {
            (PI - x, 1.0)
        }
    } else {
        if x <= FRAC_3PI_2 {
            (x - PI, -1.0)
        } else {
            (TAU - x, -1.0)
        }
    };

    sign * sine_poly(x_reduced)
}

#[inline]
pub fn approx_cosine(x: f32) -> f32 {
    approx_sine(x + FRAC_PI_2)
}

const SINE_0: f32 = 1.168518102600408e-05; // 0. coef in sine function
const SINE_1: f32 = 9.996429605292424e-01; // 1. coef in sine function
const SINE_2: f32 = 2.296591796643780e-03; // 2. coef in sine function
const SINE_3: f32 = -1.722893586265541e-01; // 3. coef in sine function
const SINE_4: f32 = 6.067090078964542e-03; // 4. coef in sine function
const SINE_5: f32 = 5.742717420336717e-03; // 5. coef in sine function
#[inline]
fn sine_poly(x: f32) -> f32 {
    SINE_0 + x * (SINE_1 + x * (SINE_2 + x * (SINE_3 + x * (SINE_4 + x * SINE_5))))
}
