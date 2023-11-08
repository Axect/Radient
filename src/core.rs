use casey::pascal;
use peroxide::traits::num::{ExpLogOps, PowOps, TrigOps};
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Default)]
pub struct Graph {
    gradients: Vec<f64>,
    value_buffer: Vec<Option<f64>>,
    nodes: Vec<Node>, // Added to store the nodes
}

pub enum Node {
    Var(usize),         // Index in the value buffer
    Add(usize, usize),  // Indices of the left and right operands
    Addf(f64, usize),
    Sub(usize, usize),
    Subf(usize, f64),
    Mul(usize, usize),
    Mulf(f64, usize),
    Div(usize, usize),
    Pow(usize, usize),
    Powf(usize, f64),
    Powi(usize, i32),
    Neg(usize),         // Index of the operand
    Recip(usize),
    Exp(usize),
    Ln(usize),
    Sin(usize),
    Cos(usize),
    Tan(usize),
    Sinh(usize),
    Cosh(usize),
    Tanh(usize),
}

macro_rules! impl_unary_op {
    ($name:ident) => {
        pub fn $name(&mut self, operand: usize) -> usize {
            let index = self.nodes.len();
            self.value_buffer.push(None);
            self.gradients.push(0.0);
            self.nodes.push(pascal!(Node::$name)(operand));
            index
        }
    };
}

macro_rules! impl_binary_op {
    ($name:ident) => {
        pub fn $name(&mut self, left: usize, right: usize) -> usize {
            let index = self.nodes.len();
            self.value_buffer.push(None);
            self.gradients.push(0.0);
            self.nodes.push(pascal!(Node::$name)(left, right));
            index
        }
    };
}

impl Graph {
    pub fn var(&mut self, value: f64) -> usize {
        let index = self.value_buffer.len();
        self.value_buffer.push(Some(value));
        self.gradients.push(0.0);
        self.nodes.push(Node::Var(index));
        index // The index is used to refer to this variable
    }

    // Implement the unary operators
    impl_unary_op!(neg);
    impl_unary_op!(recip);
    impl_unary_op!(exp);
    impl_unary_op!(ln);
    impl_unary_op!(sin);
    impl_unary_op!(cos);
    impl_unary_op!(tan);
    impl_unary_op!(sinh);
    impl_unary_op!(cosh);
    impl_unary_op!(tanh);

    // Implement the binary operators
    impl_binary_op!(add);
    impl_binary_op!(sub);
    impl_binary_op!(mul);
    impl_binary_op!(div);
    impl_binary_op!(pow);

    pub fn addf(&mut self, num: f64, right: usize) -> usize {
        let index = self.nodes.len();
        self.value_buffer.push(None);
        self.gradients.push(0.0);
        self.nodes.push(Node::Addf(num, right));
        index
    }

    pub fn subf(&mut self, left: usize, num: f64) -> usize {
        let index = self.nodes.len();
        self.value_buffer.push(None);
        self.gradients.push(0.0);
        self.nodes.push(Node::Subf(left, num));
        index
    }

    pub fn mulf(&mut self, num: f64, right: usize) -> usize {
        let index = self.nodes.len();
        self.value_buffer.push(None);
        self.gradients.push(0.0);
        self.nodes.push(Node::Mulf(num, right));
        index
    }

    pub fn powf(&mut self, operand: usize, power: f64) -> usize {
        let index = self.nodes.len();
        self.value_buffer.push(None);
        self.gradients.push(0.0);
        self.nodes.push(Node::Powf(operand, power));
        index
    }

    pub fn powi(&mut self, operand: usize, power: i32) -> usize {
        let index = self.nodes.len();
        self.value_buffer.push(None);
        self.gradients.push(0.0);
        self.nodes.push(Node::Powi(operand, power));
        index
    }

    pub fn forward(&mut self, index: usize) -> f64 {
        match self.value_buffer[index] {
            Some(value) => value,
            None => {
                let result = match self.nodes[index] {
                    Node::Var(_) => unreachable!(),
                    Node::Add(left_index, right_index) => {
                        self.forward(left_index) + self.forward(right_index)
                    }
                    Node::Sub(left_index, right_index) => {
                        self.forward(left_index) - self.forward(right_index)
                    }
                    Node::Addf(num, right_index) => num + self.forward(right_index),
                    Node::Subf(left_index, num) => self.forward(left_index) - num,
                    Node::Mul(left_index, right_index) => {
                        self.forward(left_index) * self.forward(right_index)
                    }
                    Node::Mulf(num, right_index) => num * self.forward(right_index),
                    Node::Div(left_index, right_index) => {
                        self.forward(left_index) / self.forward(right_index)
                    }
                    Node::Pow(left_index, right_index) => {
                        self.forward(left_index).powf(self.forward(right_index))
                    }
                    Node::Powf(operand_index, power) => self.forward(operand_index).powf(power),
                    Node::Powi(operand_index, power) => self.forward(operand_index).powi(power),
                    Node::Neg(operand_index) => -self.forward(operand_index),
                    Node::Recip(operand_index) => 1.0 / self.forward(operand_index),
                    Node::Exp(operand_index) => self.forward(operand_index).exp(),
                    Node::Ln(operand_index) => self.forward(operand_index).ln(),
                    Node::Sin(operand_index) => self.forward(operand_index).sin(),
                    Node::Cos(operand_index) => self.forward(operand_index).cos(),
                    Node::Tan(operand_index) => self.forward(operand_index).tan(),
                    Node::Sinh(operand_index) => self.forward(operand_index).sinh(),
                    Node::Cosh(operand_index) => self.forward(operand_index).cosh(),
                    Node::Tanh(operand_index) => self.forward(operand_index).tanh(),
                };
                self.value_buffer[index] = Some(result);
                result
            }
        }
    }

    pub fn reset_values(&mut self) {
        self.value_buffer.iter_mut().for_each(|x| *x = None);
    }

    pub fn reset_gradients(&mut self) {
        self.gradients.iter_mut().for_each(|x| *x = 0.0);
    }

    pub fn backward(&mut self, index: usize, upstream_gradient: f64) {
        match self.nodes[index] {
            Node::Var(value_index) => {
                self.gradients[value_index] += upstream_gradient;
            }
            Node::Add(left_index, right_index) => {
                self.backward(left_index, upstream_gradient);
                self.backward(right_index, upstream_gradient);
            }
            Node::Addf(_, right_index) => {
                self.backward(right_index, upstream_gradient);
            }
            Node::Sub(left_index, right_index) => {
                self.backward(left_index, upstream_gradient);
                self.backward(right_index, -upstream_gradient);
            }
            Node::Subf(left_index, _) => {
                self.backward(left_index, upstream_gradient);
            }
            Node::Mul(left_index, right_index) => {
                let left_val = self.forward(left_index);
                let right_val = self.forward(right_index);
                self.backward(left_index, right_val * upstream_gradient);
                self.backward(right_index, left_val * upstream_gradient);
            }
            Node::Mulf(num, right_index) => {
                self.backward(right_index, num * upstream_gradient);
            }
            Node::Div(left_index, right_index) => {
                let left_val = self.forward(left_index);
                let right_val = self.forward(right_index);
                self.backward(left_index, upstream_gradient / right_val);
                self.backward(
                    right_index,
                    -upstream_gradient * left_val / right_val.powi(2),
                );
            }
            Node::Pow(left_index, right_index) => {
                let left_val = self.forward(left_index);
                let right_val = self.forward(right_index);
                self.backward(
                    left_index,
                    right_val * left_val.powf(right_val - 1.0) * upstream_gradient,
                );
                self.backward(
                    right_index,
                    left_val.ln() * left_val.powf(right_val - 1.0) * upstream_gradient,
                );
            }
            Node::Powf(operand_index, power) => {
                let operand_val = self.forward(operand_index);
                self.backward(
                    operand_index,
                    power * operand_val.powf(power - 1.0) * upstream_gradient,
                )
            }
            Node::Powi(operand_index, power) => {
                let operand_val = self.forward(operand_index);
                self.backward(
                    operand_index,
                    power as f64 * operand_val.powi(power - 1) * upstream_gradient,
                )
            }
            Node::Neg(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, -upstream_gradient * operand_val);
            }
            Node::Recip(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, -upstream_gradient / operand_val.powi(2));
            }
            Node::Exp(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, operand_val.exp() * upstream_gradient);
            }
            Node::Ln(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, upstream_gradient / operand_val);
            }
            Node::Sin(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, operand_val.cos() * upstream_gradient);
            }
            Node::Cos(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, -operand_val.sin() * upstream_gradient);
            }
            Node::Tan(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(
                    operand_index,
                    (1f64 + operand_val.tan().powi(2)) * upstream_gradient,
                )
            }
            Node::Sinh(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, operand_val.cosh() * upstream_gradient);
            }
            Node::Cosh(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, operand_val.sinh() * upstream_gradient);
            }
            Node::Tanh(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(
                    operand_index,
                    (1f64 - operand_val.tanh().powi(2)) * upstream_gradient,
                )
            }
        }
    }

    pub fn get_gradient(&self, var_index: usize) -> f64 {
        self.gradients[var_index]
    }

    pub fn compile(&mut self, expr: Expr) -> usize {
        parse_expr(expr, self)
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

    fn sin_cos(&self) -> (Self, Self) {
        (
            Expr::Sin(Box::new(self.clone())),
            Expr::Cos(Box::new(self.clone())),
        )
    }

    fn sinh_cosh(&self) -> (Self, Self) {
        (
            Expr::Sinh(Box::new(self.clone())),
            Expr::Cosh(Box::new(self.clone())),
        )
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
    fn pow(&self, rhs: Self) -> Self {
        Expr::Pow(Box::new(self.clone()), Box::new(rhs))
    }

    fn powf(&self, rhs: f64) -> Self {
        Expr::Powf(Box::new(self.clone()), rhs)
    }

    fn powi(&self, rhs: i32) -> Self {
        Expr::Powi(Box::new(self.clone()), rhs)
    }
}

impl ExpLogOps for Expr {
    fn exp(&self) -> Self {
        Expr::Exp(Box::new(self.clone()))
    }

    fn ln(&self) -> Self {
        Expr::Ln(Box::new(self.clone()))
    }

    fn log(&self, _base: f64) -> Self {
        todo!()
    }
}

// ┌──────────────────────────────────────────────────────────┐
//  Parsing Expr to Graph
// └──────────────────────────────────────────────────────────┘
pub fn parse_expr(expr: Expr, graph: &mut Graph) -> usize {
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
    }
}
