use peroxide::fuga::{Matrix, diag, Shape, matrix, zeros, zeros_shape, ml_matrix, Printable};
use peroxide_num::{Group, Ring, PowOps, ExpLogOps, TrigOps, Numeric};
use std::ops::{Neg, Add, Mul, Sub, Div};
use self::MatrixRing::*;
use crate::core::ActivationFunction;

#[derive(Debug, Clone)]
pub enum MatrixRing {
    Matrix(Matrix),
    Diag(f64),
}

impl MatrixRing {
    pub fn matrix(data: Vec<f64>, row: usize, col: usize, shape: Shape) -> Self {
        MatrixRing::Matrix(matrix(data, row, col, shape))
    }

    pub fn zeros_shape(row: usize, col: usize, shape: Shape) -> Self {
        MatrixRing::Matrix(zeros_shape(row, col, shape))
    }

    pub fn zeros(row: usize, col: usize) -> Self {
        MatrixRing::Matrix(zeros(row, col))
    }

    pub fn ml_matrix(str: &str) -> Self {
        MatrixRing::Matrix(ml_matrix(str))
    }
}

impl std::fmt::Display for MatrixRing {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MatrixRing::Matrix(m) => write!(f, "{}", m),
            MatrixRing::Diag(n) => write!(f, "Diag {}", n),
        }
    }
}

impl Printable for MatrixRing {
    fn print(&self) {
        println!("{}", self);
    }
}

impl Default for MatrixRing {
    fn default() -> Self {
        Diag(0f64)
    }
}

impl Neg for MatrixRing {
    type Output = MatrixRing;

    fn neg(self) -> Self::Output {
        match self {
            MatrixRing::Matrix(m) => MatrixRing::Matrix(-m),
            MatrixRing::Diag(n) => MatrixRing::Diag(-n),
        }
    }
}

impl Add for MatrixRing {
    type Output = MatrixRing;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Diag(x), Diag(y)) => Diag(x + y),
            (Matrix(m1), Matrix(m2)) => Matrix(m1 + m2),
            (Diag(x), m) if x == 0f64 => m,
            (m, Diag(x)) if x == 0f64 => m,
            (Diag(x), Matrix(m)) => Matrix(&m + &(diag(m.col) * x)),
            (Matrix(m), Diag(x)) => Matrix(&m + &(diag(m.col) * x)),
        }
    }
}

impl Add<f64> for MatrixRing {
    type Output = MatrixRing;

    fn add(self, rhs: f64) -> Self::Output {
        match self {
            Matrix(m) => Matrix(m + rhs),
            _ => panic!("unsupported operation"),
        }
    }
}

impl Group for MatrixRing {
    fn zero() -> Self {
        Diag(0f64)
    }
}

impl Sub for MatrixRing {
    type Output = MatrixRing;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Diag(x), Diag(y)) => Diag(x - y),
            (Matrix(m1), Matrix(m2)) => Matrix(m1 - m2),
            (Matrix(m), Diag(x)) => Matrix(&m - &(diag(m.col) * x)),
            (Diag(x), Matrix(m)) => Matrix(&(diag(m.col) * x) - &m),
        }
    }
}

impl Sub<f64> for MatrixRing {
    type Output = MatrixRing;

    fn sub(self, rhs: f64) -> Self::Output {
        match self {
            Matrix(m) => Matrix(m - rhs),
            _ => panic!("unsupported operation"),
        }
    }
}

impl Mul for MatrixRing {
    type Output = MatrixRing;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Diag(x), Diag(y)) => Diag(x * y),
            (Matrix(m1), Matrix(m2)) => Matrix(m1 * m2),
            (Diag(x), Matrix(m)) => Matrix(m * x),
            (Matrix(m), Diag(x)) => Matrix(m * x),
        }
    }
}

impl Mul<f64> for MatrixRing {
    type Output = MatrixRing;

    fn mul(self, rhs: f64) -> Self::Output {
        match self {
            Matrix(m) => Matrix(m * rhs),
            Diag(x) => Diag(x * rhs),
        }
    }
}

impl Ring for MatrixRing {
    fn one() -> Self {
        Diag(1f64)
    }
}

impl Div for MatrixRing {
    type Output = MatrixRing;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Diag(x), Diag(y)) => Diag(x / y),
            (Matrix(m1), Matrix(m2)) => Matrix(m1 / m2),
            (Matrix(m), Diag(x)) => Matrix(m / x),
            (Diag(x), Matrix(m)) => Matrix(x / m),
        }
    }
}

impl Div<f64> for MatrixRing {
    type Output = MatrixRing;

    fn div(self, rhs: f64) -> Self::Output {
        match self {
            Matrix(m) => Matrix(m / rhs),
            Diag(x) => Diag(x / rhs),
        }
    }
}

impl Add<MatrixRing> for f64 {
    type Output = MatrixRing;

    fn add(self, rhs: MatrixRing) -> Self::Output {
        match rhs {
            Matrix(m) => Matrix(m + self),
            Diag(x) => Diag(x + self),
        }
    }
}

impl Sub<MatrixRing> for f64 {
    type Output = MatrixRing;

    fn sub(self, rhs: MatrixRing) -> Self::Output {
        match rhs {
            Matrix(m) => Matrix(self - m),
            Diag(x) => Diag(self - x),
        }
    }
}

impl Mul<MatrixRing> for f64 {
    type Output = MatrixRing;

    fn mul(self, rhs: MatrixRing) -> Self::Output {
        match rhs {
            Matrix(m) => Matrix(self * m),
            Diag(x) => Diag(self * x),
        }
    }
}

impl Div<MatrixRing> for f64 {
    type Output = MatrixRing;

    fn div(self, rhs: MatrixRing) -> Self::Output {
        match rhs {
            Matrix(m) => Matrix(self / m),
            Diag(x) => Diag(self / x),
        }
    }
}

macro_rules! impl_unary_ops {
    ($op:ident) => {
        fn $op(&self) -> Self {
            match self {
                Matrix(m) => Matrix(m.$op()),
                Diag(x) => Diag(x.$op()),
            }
        }
    }
}

impl PowOps for MatrixRing {
    type Float = f64;

    fn powi(&self, rhs: i32) -> Self {
        match self {
            Matrix(m) => Matrix(m.powi(rhs)),
            Diag(x) => Diag(x.powi(rhs)),
        }
    }

    fn powf(&self, rhs: f64) -> Self {
        match self {
            Matrix(m) => Matrix(m.powf(rhs)),
            Diag(x) => Diag(x.powf(rhs)),
        }
    }

    fn pow(&self, _rhs: Self) -> Self {
        unimplemented!()
    }

    impl_unary_ops!(sqrt);
}

impl ExpLogOps for MatrixRing {
    type Float = f64;

    impl_unary_ops!(exp);
    impl_unary_ops!(ln);
    impl_unary_ops!(log2);
    impl_unary_ops!(log10);

    fn log(&self, rhs: f64) -> Self {
        match self {
            Matrix(m) => Matrix(m.log(rhs)),
            Diag(x) => Diag(x.log(rhs)),
        }
    }
}


impl TrigOps for MatrixRing {
    impl_unary_ops!(sin);
    impl_unary_ops!(cos);
    impl_unary_ops!(tan);
    impl_unary_ops!(asin);
    impl_unary_ops!(acos);
    impl_unary_ops!(atan);
    impl_unary_ops!(sinh);
    impl_unary_ops!(cosh);
    impl_unary_ops!(tanh);
    impl_unary_ops!(asinh);
    impl_unary_ops!(acosh);
    impl_unary_ops!(atanh);

    fn sin_cos(&self) -> (Self, Self) {
        match self {
            Matrix(m) => {
                let (sin, cos) = m.sin_cos();
                (Matrix(sin), Matrix(cos))
            }
            Diag(x) => {
                let (sin, cos) = x.sin_cos();
                (Diag(sin), Diag(cos))
            }
        }
    }
}

impl Numeric<f64> for MatrixRing {}

impl ActivationFunction for MatrixRing {
    fn sigmoid(&self) -> Self {
        match self {
            Matrix(m) => Matrix(m.sigmoid()),
            Diag(x) => Diag(x.sigmoid()),
        }
    }

    fn relu(&self) -> Self {
        match self {
            Matrix(m) => Matrix(m.relu()),
            Diag(x) => Diag(x.relu()),
        }
    }

    fn heaviside_zero(&self) -> Self {
        match self {
            Matrix(m) => Matrix(m.heaviside_zero()),
            Diag(x) => Diag(x.heaviside_zero()),
        }
    }
}
