use std::f32::consts::{FRAC_PI_2, PI, TAU};
const FRAC_3PI_2: f32 = PI + FRAC_PI_2;

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    const S0: f32 = 6.549889643289383e-06;
    const S1: f32 = 2.472077354175178e-01;
    const S2: f32 = 1.290223762394391e-02;
    const S3: f32 = -4.076700256613482e-02;
    const S4: f32 = 1.403104754073916e-02;
    const S5: f32 = -2.408918337689537e-03;
    const S6: f32 = 2.317087210268684e-04;
    const S7: f32 = -1.192241369934327e-05;
    const S8: f32 = 2.560170922345861e-07;

    if x >= 10.0 {
        1.0
    } else if x <= -10.0 {
        0.0
    } else {
        let (sgn, x) = (x.signum(), x.abs());
        0.5 + sgn
            * (S0
                + x * (S1
                    + x * (S2 + x * (S3 + x * (S4 + x * (S5 + x * (S6 + x * (S7 + x * S8))))))))
    }
}

#[inline]
pub fn approx_sine(mut x: f32) -> f32 {
    x = x.rem_euclid(TAU);

    let (x_reduced, sgn) = if x <= PI {
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

    sgn * sine_poly(x_reduced)
}

#[inline]
pub fn approx_cosine(mut x: f32) -> f32 {
    x = x.rem_euclid(TAU);

    let (x_reduced, sgn) = if x <= PI {
        if x <= FRAC_PI_2 {
            (FRAC_PI_2 - x, 1.0)
        } else {
            (x - FRAC_PI_2, -1.0)
        }
    } else {
        if x <= FRAC_3PI_2 {
            (FRAC_3PI_2 - x, -1.0)
        } else {
            (x - FRAC_3PI_2, 1.0)
        }
    };

    sgn * sine_poly(x_reduced)
}
#[inline]
fn sine_poly(x: f32) -> f32 {
    const SINE_0: f32 = 1.168_518_1e-05;
    const SINE_1: f32 = 9.996_429_7e-01;
    const SINE_2: f32 = 2.296_591_9e-03;
    const SINE_3: f32 = -1.722_893_6e-01;
    const SINE_4: f32 = 6.067_09e-03;
    const SINE_5: f32 = 5.742_717_5e-03;
    SINE_0 + x * (SINE_1 + x * (SINE_2 + x * (SINE_3 + x * (SINE_4 + x * SINE_5))))
}
