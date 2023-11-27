use casey::pascal;
use peroxide_num::{ExpLogOps, Numeric, PowOps, TrigOps};
use std::ops::{Add, Div, Mul, Neg, Sub};
use crate::traits::{ActivationFunction, Matrizable};

#[derive(Default)]
pub struct Graph<T> {
    gradients: Vec<T>,
    buffer: Vec<Option<T>>,
    nodes: Vec<Node>, // Added to store the nodes
    value_ics: Vec<usize>,
    compiled: Option<usize>,
}

pub enum Node {
    Var(usize),        // Index in the value buffer
    Add(usize, usize), // Indices of the left and right operands
    Addf(f64, usize),
    Sub(usize, usize),
    Subf(usize, f64),
    Mul(usize, usize),
    Mulf(f64, usize),
    Hadamard(usize, usize),
    Transpose(usize),
    Div(usize, usize),
    Pow(usize, usize),
    Powf(usize, f64),
    Powi(usize, i32),
    Neg(usize), // Index of the operand
    Recip(usize),
    Exp(usize),
    Ln(usize),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    Sinh(usize),
    Cosh(usize),
    Tanh(usize),
    Sigmoid(usize),
    ReLU(usize),
}

macro_rules! impl_unary_op {
    ($name:ident, $t:ty) => {
        pub fn $name(&mut self, operand: usize) -> usize {
            let index = self.nodes.len();
            self.buffer.push(None);
            self.gradients.push(<$t>::default());
            self.nodes.push(pascal!(Node::$name)(operand));
            index
        }
    };
}

macro_rules! impl_binary_op {
    ($name:ident, $t:ty) => {
        pub fn $name(&mut self, left: usize, right: usize) -> usize {
            let index = self.nodes.len();
            self.buffer.push(None);
            self.gradients.push(<$t>::default());
            self.nodes.push(pascal!(Node::$name)(left, right));
            index
        }
    };
}

impl<T: std::fmt::Debug + Numeric<f64> + Default + ActivationFunction + Matrizable> Graph<T>
where
    f64: Div<T, Output = T>,
{
    pub fn var(&mut self, value: T) -> usize {
        let index = self.buffer.len();
        self.gradients.push(value.zeros_like());
        self.buffer.push(Some(value));
        self.nodes.push(Node::Var(index));
        self.value_ics.push(index);
        index // The index is used to refer to this variable
    }

    /// Declare n_vars variables (But not initialize them)
    pub fn touch_vars(&mut self, n_vars: usize) {
        let start_index = self.buffer.len();
        self.buffer.resize(start_index + n_vars, None);
        self.gradients.resize(start_index + n_vars, T::default());
        for i in 0..n_vars {
            self.nodes.push(Node::Var(start_index + i));
            self.value_ics.push(start_index + i);
        }
    }

    pub fn get_var(&self, var_order: usize) -> usize {
        self.value_ics[var_order]
    }

    pub fn get_vars(&self) -> Vec<usize> {
        self.value_ics.clone()
    }

    pub fn get_values(&self) -> Vec<Option<T>> {
        self.buffer.clone()
    }

    pub fn subs_var(&mut self, index: usize, value: T) {
        self.gradients[index] = value.zeros_like();
        self.buffer[index] = Some(value);
    }

    pub fn subs_vars(&mut self, vals: &[T]) {
        let value_ics = &self.value_ics;
        assert!(value_ics.len() >= vals.len());

        for (i, val) in value_ics.iter().zip(vals) {
            self.buffer[*i] = Some(val.clone());
            self.gradients[*i] = val.zeros_like();
        }
    }

    pub fn get_symbol(&self, var_order: usize) -> Expr {
        Expr::Symbol(self.get_var(var_order))
    }

    pub fn get_symbols(&self) -> Vec<Expr> {
        self.get_vars()
            .iter()
            .map(|x| Expr::Symbol(*x))
            .collect::<Vec<_>>()
    }

    // Implement the unary operators
    impl_unary_op!(neg, T);
    impl_unary_op!(recip, T);
    impl_unary_op!(exp, T);
    impl_unary_op!(ln, T);
    impl_unary_op!(sin, T);
    impl_unary_op!(cos, T);
    impl_unary_op!(tan, T);
    impl_unary_op!(sinh, T);
    impl_unary_op!(cosh, T);
    impl_unary_op!(tanh, T);
    impl_unary_op!(sigmoid, T);
    impl_unary_op!(transpose, T);

    // Implement the binary operators
    impl_binary_op!(add, T);
    impl_binary_op!(sub, T);
    impl_binary_op!(mul, T);
    impl_binary_op!(div, T);
    impl_binary_op!(pow, T);
    impl_binary_op!(hadamard, T);

    pub fn addf(&mut self, num: f64, right: usize) -> usize {
        let index = self.nodes.len();
        self.buffer.push(None);
        self.gradients.push(T::default());
        self.nodes.push(Node::Addf(num, right));
        index
    }

    pub fn subf(&mut self, left: usize, num: f64) -> usize {
        let index = self.nodes.len();
        self.buffer.push(None);
        self.gradients.push(T::default());
        self.nodes.push(Node::Subf(left, num));
        index
    }

    pub fn mulf(&mut self, num: f64, right: usize) -> usize {
        let index = self.nodes.len();
        self.buffer.push(None);
        self.gradients.push(T::default());
        self.nodes.push(Node::Mulf(num, right));
        index
    }

    pub fn powf(&mut self, operand: usize, power: f64) -> usize {
        let index = self.nodes.len();
        self.buffer.push(None);
        self.gradients.push(T::default());
        self.nodes.push(Node::Powf(operand, power));
        index
    }

    pub fn powi(&mut self, operand: usize, power: i32) -> usize {
        let index = self.nodes.len();
        self.buffer.push(None);
        self.gradients.push(T::default());
        self.nodes.push(Node::Powi(operand, power));
        index
    }

    pub fn relu(&mut self, operand: usize) -> usize {
        let index = self.nodes.len();
        self.buffer.push(None);
        self.gradients.push(T::default());
        self.nodes.push(Node::ReLU(operand));
        index
    }

    pub fn forward_step(&mut self, index: usize) -> T {
        match &self.buffer[index] {
            Some(value) => value.clone(),
            None => {
                let result = match self.nodes[index] {
                    Node::Var(_) => unreachable!(),
                    Node::Add(left_index, right_index) => {
                        self.forward_step(left_index) + self.forward_step(right_index)
                    }
                    Node::Sub(left_index, right_index) => {
                        self.forward_step(left_index) - self.forward_step(right_index)
                    }
                    Node::Addf(num, right_index) => self.forward_step(right_index) + num,
                    Node::Subf(left_index, num) => self.forward_step(left_index) - num,
                    Node::Mul(left_index, right_index) => {
                        self.forward_step(left_index) * self.forward_step(right_index)
                    }
                    Node::Mulf(num, right_index) => self.forward_step(right_index) * num,
                    Node::Hadamard(left_index, right_index) => {
                        self.forward_step(left_index).hadamard(&self.forward_step(right_index))
                    }
                    Node::Transpose(operand_index) => {
                        self.forward_step(operand_index).transpose()
                    }
                    Node::Div(left_index, right_index) => {
                        self.forward_step(left_index) / self.forward_step(right_index)
                    }
                    Node::Pow(left_index, right_index) => self
                        .forward_step(left_index)
                        .pow(self.forward_step(right_index)),
                    Node::Powf(operand_index, power) => {
                        self.forward_step(operand_index).powf(power)
                    }
                    Node::Powi(operand_index, power) => {
                        self.forward_step(operand_index).powi(power)
                    }
                    Node::Neg(operand_index) => -self.forward_step(operand_index),
                    Node::Recip(operand_index) => 1.0 / self.forward_step(operand_index),
                    Node::Exp(operand_index) => self.forward_step(operand_index).exp(),
                    Node::Ln(operand_index) => self.forward_step(operand_index).ln(),
                    Node::Sin(operand_index) => self.forward_step(operand_index).sin(),
                    Node::Cos(operand_index) => self.forward_step(operand_index).cos(),
                    Node::Tan(operand_index) => self.forward_step(operand_index).tan(),
                    Node::Sinh(operand_index) => self.forward_step(operand_index).sinh(),
                    Node::Cosh(operand_index) => self.forward_step(operand_index).cosh(),
                    Node::Tanh(operand_index) => self.forward_step(operand_index).tanh(),
                    Node::Sigmoid(operand_index) => self.forward_step(operand_index).sigmoid(),
                    Node::ReLU(operand_index) => self.forward_step(operand_index).relu(),
                };
                self.buffer[index] = Some(result.clone());
                result
            }
        }
    }

    /// Reset values & gradients without variables
    pub fn reset(&mut self) {
        let except_ics = &self.value_ics;
        let reset_ics = (0..self.buffer.len()).filter(|x| !except_ics.contains(x));

        for i in reset_ics {
            self.buffer[i] = None;
            self.gradients[i] = T::default();
        }

        for i in except_ics {
            self.gradients[*i] = T::default();
        }
    }

    #[allow(unused_variables)]
    pub fn backward_step(&mut self, index: usize, upstream_gradient: T) {
        match self.nodes[index] {
            Node::Var(value_index) => {
                self.gradients[value_index] =
                    self.gradients[value_index].clone() + upstream_gradient;
            }
            Node::Add(left_index, right_index) => {
                self.backward_step(left_index, upstream_gradient.clone());
                self.backward_step(right_index, upstream_gradient);
            }
            Node::Addf(_, right_index) => {
                self.backward_step(right_index, upstream_gradient);
            }
            Node::Sub(left_index, right_index) => {
                self.backward_step(left_index, upstream_gradient.clone());
                self.backward_step(right_index, -upstream_gradient);
            }
            Node::Subf(left_index, _) => {
                self.backward_step(left_index, upstream_gradient);
            }
            Node::Mul(left_index, right_index) => {
                let left_val = self.forward_step(left_index);
                let right_val = self.forward_step(right_index);
                self.backward_step(left_index, upstream_gradient.clone() * right_val.transpose());
                self.backward_step(right_index, left_val.transpose() * upstream_gradient);
            }
            Node::Mulf(num, right_index) => {
                self.backward_step(right_index, upstream_gradient * num);
            }
            Node::Hadamard(left_index, right_index) => {
                let left_val = self.forward_step(left_index);
                let right_val = self.forward_step(right_index);
                self.backward_step(
                    left_index,
                    right_val.hadamard(&upstream_gradient),
                );
                self.backward_step(
                    right_index,
                    left_val.hadamard(&upstream_gradient),
                );
            }
            Node::Transpose(operand_index) => {
                self.backward_step(operand_index, upstream_gradient.transpose());
            }
            Node::Div(left_index, right_index) => {
                let left_val = self.forward_step(left_index);
                let right_val = self.forward_step(right_index);
                self.backward_step(left_index, upstream_gradient.clone() / right_val.clone());
                self.backward_step(
                    right_index,
                    -(left_val / right_val.hadamard(&right_val)).hadamard(&upstream_gradient),
                );
            }
            Node::Pow(left_index, right_index) => {
                todo!()
            }
            Node::Powf(operand_index, power) => {
                todo!()
            }
            Node::Powi(operand_index, power) => {
                todo!()
            }
            Node::Neg(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, -operand_val.hadamard(&upstream_gradient));
            }
            Node::Recip(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, -upstream_gradient / (operand_val.hadamard(&operand_val)));
            }
            Node::Exp(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, operand_val.exp().hadamard(&upstream_gradient));
            }
            Node::Ln(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, upstream_gradient / operand_val);
            }
            Node::Sin(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, operand_val.cos().hadamard(&upstream_gradient));
            }
            Node::Cos(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, -operand_val.sin().hadamard(&upstream_gradient));
            }
            Node::Tan(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                let tan = operand_val.tan();
                self.backward_step(
                    operand_index,
                    (tan.hadamard(&tan) + 1f64).hadamard(&upstream_gradient),
                )
            }
            Node::Sinh(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, operand_val.cosh().hadamard(&upstream_gradient));
            }
            Node::Cosh(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                self.backward_step(operand_index, operand_val.sinh().hadamard(&upstream_gradient));
            }
            Node::Tanh(operand_index) => {
                let operand_val = self.forward_step(operand_index);
                let tanh = operand_val.tanh();
                self.backward_step(
                    operand_index,
                    (-(tanh.hadamard(&tanh) - 1f64)).hadamard(&upstream_gradient),
                );
            }
            Node::Sigmoid(operand_index) => {
                let operand_val = self.forward_step(operand_index).sigmoid();
                let diff_from_one = -operand_val.clone() + 1f64;
                self.backward_step(
                    operand_index,
                    operand_val.hadamard(&diff_from_one).hadamard(&upstream_gradient),
                );
            }
            Node::ReLU(operand_index) => {
                let operand_val = self.forward_step(operand_index).heaviside_zero();
                self.backward_step(operand_index, operand_val.hadamard(&upstream_gradient));
            }
        }
    }

    pub fn get_gradient(&self, index: usize) -> T {
        self.gradients[index].clone()
    }

    pub fn get_gradients(&self) -> Vec<T> {
        let value_ics = self.get_vars();
        value_ics.iter().map(|x| self.get_gradient(*x)).collect()
    }

    pub fn compile(&mut self, expr: Expr) {
        self.compiled = Some(parse_expr(expr, self))
    }

    pub fn get_compiled(&self) -> Option<usize> {
        self.compiled
    }

    pub fn forward(&mut self) -> T {
        match self.compiled {
            Some(idx) => self.forward_step(idx),
            None => panic!("No compiled expression"),
        }
    }

    pub fn backward(&mut self) {
        match self.compiled {
            Some(idx) => {
                let value = self.buffer[idx].as_ref().unwrap().ones_like();
                println!("Value: {:?}", value);
                self.backward_step(idx, value);
            },
            None => panic!("No compiled expression"),
        }
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Symbol for generating Abstract Expressions
// └──────────────────────────────────────────────────────────┘
#[derive(Debug, Clone)]
pub enum Expr {
    Symbol(usize),
    Add(Box<Expr>, Box<Expr>),
    Addf(f64, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Subf(Box<Expr>, f64),
    Mul(Box<Expr>, Box<Expr>),
    Mulf(f64, Box<Expr>),
    Hadamard(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Pow(Box<Expr>, Box<Expr>),
    Powf(Box<Expr>, f64),
    Powi(Box<Expr>, i32),
    Neg(Box<Expr>),
    Recip(Box<Expr>),
    Exp(Box<Expr>),
    Ln(Box<Expr>),
    Sin(Box<Expr>),
    Cos(Box<Expr>),
    Tan(Box<Expr>),
    Sinh(Box<Expr>),
    Cosh(Box<Expr>),
    Tanh(Box<Expr>),
    Sigmoid(Box<Expr>),
    ReLU(Box<Expr>),
}

impl Neg for Expr {
    type Output = Expr;

    fn neg(self) -> Self::Output {
        Expr::Neg(Box::new(self))
    }
}

impl Neg for &Expr {
    type Output = Expr;

    fn neg(self) -> Self::Output {
        Expr::Neg(Box::new(self.clone()))
    }
}

impl Add for Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        Expr::Add(Box::new(self), Box::new(rhs))
    }
}

impl Add for &Expr {
    type Output = Expr;

    fn add(self, rhs: Self) -> Self::Output {
        Expr::Add(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Add<Expr> for f64 {
    type Output = Expr;

    fn add(self, rhs: Expr) -> Self::Output {
        Expr::Addf(self, Box::new(rhs))
    }
}

impl Add<f64> for Expr {
    type Output = Expr;

    fn add(self, rhs: f64) -> Self::Output {
        Expr::Addf(rhs, Box::new(self))
    }
}

impl Add<&Expr> for f64 {
    type Output = Expr;

    fn add(self, rhs: &Expr) -> Self::Output {
        Expr::Addf(self, Box::new(rhs.clone()))
    }
}

impl Add<f64> for &Expr {
    type Output = Expr;

    fn add(self, rhs: f64) -> Self::Output {
        Expr::Addf(rhs, Box::new(self.clone()))
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, rhs: Self) -> Self::Output {
        Expr::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Sub for &Expr {
    type Output = Expr;

    fn sub(self, rhs: Self) -> Self::Output {
        Expr::Sub(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Sub<Expr> for f64 {
    type Output = Expr;

    fn sub(self, rhs: Expr) -> Self::Output {
        Expr::Neg(Box::new(Expr::Subf(Box::new(rhs), self)))
    }
}

impl Sub<f64> for Expr {
    type Output = Expr;

    fn sub(self, rhs: f64) -> Self::Output {
        Expr::Subf(Box::new(self), rhs)
    }
}

impl Sub<&Expr> for f64 {
    type Output = Expr;

    fn sub(self, rhs: &Expr) -> Self::Output {
        Expr::Subf(Box::new(rhs.clone()), self)
    }
}

impl Sub<f64> for &Expr {
    type Output = Expr;

    fn sub(self, rhs: f64) -> Self::Output {
        Expr::Subf(Box::new(self.clone()), rhs)
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        Expr::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Mul for &Expr {
    type Output = Expr;

    fn mul(self, rhs: Self) -> Self::Output {
        Expr::Mul(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Mul<Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: Expr) -> Self::Output {
        Expr::Mulf(self, Box::new(rhs))
    }
}

impl Mul<f64> for Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Self::Output {
        Expr::Mulf(rhs, Box::new(self))
    }
}

impl Mul<&Expr> for f64 {
    type Output = Expr;

    fn mul(self, rhs: &Expr) -> Self::Output {
        Expr::Mulf(self, Box::new(rhs.clone()))
    }
}

impl Mul<f64> for &Expr {
    type Output = Expr;

    fn mul(self, rhs: f64) -> Self::Output {
        Expr::Mulf(rhs, Box::new(self.clone()))
    }
}

impl Div for Expr {
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        Expr::Div(Box::new(self), Box::new(rhs))
    }
}

impl Div for &Expr {
    type Output = Expr;

    fn div(self, rhs: Self) -> Self::Output {
        Expr::Div(Box::new(self.clone()), Box::new(rhs.clone()))
    }
}

impl Div<Expr> for f64 {
    type Output = Expr;

    fn div(self, rhs: Expr) -> Self::Output {
        Expr::Recip(Box::new(rhs))
    }
}

impl Div<f64> for Expr {
    type Output = Expr;

    fn div(self, rhs: f64) -> Self::Output {
        self.mul(rhs.recip())
    }
}

impl Div<&Expr> for f64 {
    type Output = Expr;

    fn div(self, rhs: &Expr) -> Self::Output {
        Expr::Recip(Box::new(rhs.clone()))
    }
}

impl Div<f64> for &Expr {
    type Output = Expr;

    fn div(self, rhs: f64) -> Self::Output {
        self.mul(rhs.recip())
    }
}

impl TrigOps for Expr {
    fn sin_cos(&self) -> (Self, Self) {
        (
            Expr::Sin(Box::new(self.clone())),
            Expr::Cos(Box::new(self.clone())),
        )
    }

    fn sin(&self) -> Self {
        Expr::Sin(Box::new(self.clone()))
    }

    fn cos(&self) -> Self {
        Expr::Cos(Box::new(self.clone()))
    }

    fn tan(&self) -> Self {
        Expr::Tan(Box::new(self.clone()))
    }

    fn sinh(&self) -> Self {
        Expr::Sinh(Box::new(self.clone()))
    }

    fn cosh(&self) -> Self {
        Expr::Cosh(Box::new(self.clone()))
    }

    fn tanh(&self) -> Self {
        Expr::Tanh(Box::new(self.clone()))
    }

    fn asin(&self) -> Self {
        todo!()
    }

    fn acos(&self) -> Self {
        todo!()
    }

    fn atan(&self) -> Self {
        todo!()
    }

    fn asinh(&self) -> Self {
        todo!()
    }

    fn acosh(&self) -> Self {
        todo!()
    }

    fn atanh(&self) -> Self {
        todo!()
    }
}

impl PowOps for Expr {
    type Float = f64;

    fn powi(&self, rhs: i32) -> Self {
        Expr::Powi(Box::new(self.clone()), rhs)
    }

    fn powf(&self, rhs: f64) -> Self {
        Expr::Powf(Box::new(self.clone()), rhs)
    }

    fn pow(&self, rhs: Self) -> Self {
        Expr::Pow(Box::new(self.clone()), Box::new(rhs))
    }

    fn sqrt(&self) -> Self {
        Expr::Powf(Box::new(self.clone()), 0.5)
    }
}

impl ExpLogOps for Expr {
    type Float = f64;

    fn exp(&self) -> Self {
        Expr::Exp(Box::new(self.clone()))
    }

    fn ln(&self) -> Self {
        Expr::Ln(Box::new(self.clone()))
    }

    fn log(&self, _base: f64) -> Self {
        todo!()
    }

    fn log2(&self) -> Self {
        todo!()
    }

    fn log10(&self) -> Self {
        todo!()
    }
}

impl Numeric<f64> for Expr {}

// ┌──────────────────────────────────────────────────────────┐
//  Parsing Expr to Graph
// └──────────────────────────────────────────────────────────┘
pub fn parse_expr<T: std::fmt::Debug + Numeric<f64> + Default + ActivationFunction + Matrizable>(
    expr: Expr,
    graph: &mut Graph<T>,
) -> usize
where
    f64: Div<T, Output = T>,
{
    match expr {
        Expr::Symbol(index) => index,
        Expr::Add(left, right) => {
            let left_index = parse_expr(*left, graph);
            let right_index = parse_expr(*right, graph);
            graph.add(left_index, right_index)
        }
        Expr::Addf(num, right) => {
            let right_index = parse_expr(*right, graph);
            graph.addf(num, right_index)
        }
        Expr::Sub(left, right) => {
            let left_index = parse_expr(*left, graph);
            let right_index = parse_expr(*right, graph);
            graph.sub(left_index, right_index)
        }
        Expr::Subf(left, num) => {
            let left_index = parse_expr(*left, graph);
            graph.subf(left_index, num)
        }
        Expr::Mul(left, right) => {
            let left_index = parse_expr(*left, graph);
            let right_index = parse_expr(*right, graph);
            graph.mul(left_index, right_index)
        }
        Expr::Mulf(num, right) => {
            let right_index = parse_expr(*right, graph);
            graph.mulf(num, right_index)
        }
        Expr::Hadamard(left, right) => {
            let left_index = parse_expr(*left, graph);
            let right_index = parse_expr(*right, graph);
            graph.hadamard(left_index, right_index)
        }
        Expr::Div(left, right) => {
            let left_index = parse_expr(*left, graph);
            let right_index = parse_expr(*right, graph);
            graph.div(left_index, right_index)
        }
        Expr::Pow(left, right) => {
            let left_index = parse_expr(*left, graph);
            let right_index = parse_expr(*right, graph);
            graph.pow(left_index, right_index)
        }
        Expr::Powf(left, right) => {
            let left_index = parse_expr(*left, graph);
            graph.powf(left_index, right)
        }
        Expr::Powi(left, right) => {
            let left_index = parse_expr(*left, graph);
            graph.powi(left_index, right)
        }
        Expr::Neg(expr) => {
            let index = parse_expr(*expr, graph);
            graph.neg(index)
        }
        Expr::Recip(expr) => {
            let index = parse_expr(*expr, graph);
            graph.recip(index)
        }
        Expr::Exp(expr) => {
            let index = parse_expr(*expr, graph);
            graph.exp(index)
        }
        Expr::Ln(expr) => {
            let index = parse_expr(*expr, graph);
            graph.ln(index)
        }
        Expr::Sin(expr) => {
            let index = parse_expr(*expr, graph);
            graph.sin(index)
        }
        Expr::Cos(expr) => {
            let index = parse_expr(*expr, graph);
            graph.cos(index)
        }
        Expr::Tan(expr) => {
            let index = parse_expr(*expr, graph);
            graph.tan(index)
        }
        Expr::Sinh(expr) => {
            let index = parse_expr(*expr, graph);
            graph.sinh(index)
        }
        Expr::Cosh(expr) => {
            let index = parse_expr(*expr, graph);
            graph.cosh(index)
        }
        Expr::Tanh(expr) => {
            let index = parse_expr(*expr, graph);
            graph.tanh(index)
        }
        Expr::Sigmoid(expr) => {
            let index = parse_expr(*expr, graph);
            graph.sigmoid(index)
        }
        Expr::ReLU(expr) => {
            let index = parse_expr(*expr, graph);
            graph.relu(index)
        }
    }
}

impl std::iter::Sum for Expr {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|a, b| Expr::Add(Box::new(a), Box::new(b)))
            .unwrap()
    }
}
