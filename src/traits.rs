use peroxide::fuga::{Matrix, matrix, FPMatrix};
use crate::core::Expr;

pub trait Matrizable {
    fn hadamard(&self, rhs: &Self) -> Self;
    fn transpose(&self) -> Self;
    fn ones_like(&self) -> Self;

    fn zeros_like(&self) -> Self;
}

impl Matrizable for f64 {
    fn hadamard(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn transpose(&self) -> Self {
        *self
    }

    fn ones_like(&self) -> Self {
        1.0
    }

    fn zeros_like(&self) -> Self {
        0.0
    }
}

impl Matrizable for Matrix {
    fn hadamard(&self, rhs: &Self) -> Self {
        self.zip_with(|x, y| x * y, rhs)
    }

    fn transpose(&self) -> Self {
        self.transpose()
    }

    fn ones_like(&self) -> Self {
        matrix(vec![1.0; self.row * self.col], self.row, self.col, self.shape)
    }

    fn zeros_like(&self) -> Self {
        matrix(vec![0.0; self.row * self.col], self.row, self.col, self.shape)
    }
}

pub trait ActivationFunction {
    fn sigmoid(&self) -> Self;
    fn relu(&self) -> Self;
    fn heaviside_zero(&self) -> Self;
}

impl ActivationFunction for f64 {
    fn sigmoid(&self) -> Self {
        1.0 / (1.0 + (-self).exp())
    }

    fn relu(&self) -> Self {
        self.max(0.0)
    }

    fn heaviside_zero(&self) -> Self {
        if self.is_sign_positive() {
            1.0
        } else {
            0.0
        }
    }
}

impl ActivationFunction for Expr {
    fn sigmoid(&self) -> Self {
        Expr::Sigmoid(Box::new(self.clone()))
    }

    fn relu(&self) -> Self {
        Expr::ReLU(Box::new(self.clone()))
    }

    fn heaviside_zero(&self) -> Self {
        unimplemented!()
    }
}

impl ActivationFunction for Matrix {
    fn sigmoid(&self) -> Self {
        let data = self.data.iter().map(|x| x.sigmoid()).collect();
        matrix(data, self.row, self.col, self.shape)
    }

    fn relu(&self) -> Self {
        let data = self.data.iter().map(|x| x.relu()).collect();
        matrix(data, self.row, self.col, self.shape)
    }

    fn heaviside_zero(&self) -> Self {
        let data = self.data.iter().map(|x| x.heaviside_zero()).collect();
        matrix(data, self.row, self.col, self.shape)
    }
}