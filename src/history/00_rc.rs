use std::cell::RefCell;
use std::rc::Rc;

pub trait ComputationalGraph {
    fn forward(&self) -> f64;
    fn backward(&self, upstream_gradient: f64);
}

#[derive(Clone)]
pub struct Var {
    value: RefCell<f64>,
    grad: RefCell<f64>,
}

impl Var {
    pub fn new(value: f64) -> Rc<Self> {
        Rc::new(Var {
            value: RefCell::new(value),
            grad: RefCell::new(0.0),
        })
    }

    pub fn gradient(&self) -> f64 {
        *self.grad.borrow()
    }
}

// Implement ComputationalGraph for Var directly, without Rc
impl ComputationalGraph for Var {
    fn forward(&self) -> f64 {
        *self.value.borrow()
    }

    fn backward(&self, upstream_gradient: f64) {
        *self.grad.borrow_mut() += upstream_gradient;
    }
}

impl ComputationalGraph for Rc<Var> {
    fn forward(&self) -> f64 {
        *self.value.borrow()
    }

    fn backward(&self, upstream_gradient: f64) {
        *self.grad.borrow_mut() += upstream_gradient;
    }
}

pub struct MultiplyOp {
    left: Rc<dyn ComputationalGraph>,
    right: Rc<dyn ComputationalGraph>,
}

impl MultiplyOp {
    pub fn new(left: Rc<dyn ComputationalGraph>, right: Rc<dyn ComputationalGraph>) -> Self {
        MultiplyOp { left, right }
    }
}

impl ComputationalGraph for MultiplyOp {
    fn forward(&self) -> f64 {
        self.left.forward() * self.right.forward()
    }

    fn backward(&self, upstream_gradient: f64) {
        let left_value = self.left.forward();
        let right_value = self.right.forward();
        let left_grad = right_value * upstream_gradient;
        let right_grad = left_value * upstream_gradient;
        self.left.backward(left_grad);
        self.right.backward(right_grad);
    }
}

#[derive(Clone)]
pub struct AddOp {
    left: Rc<dyn ComputationalGraph>,
    right: Rc<dyn ComputationalGraph>,
}

impl AddOp {
    pub fn new(left: Rc<dyn ComputationalGraph>, right: Rc<dyn ComputationalGraph>) -> Rc<Self> {
        Rc::new(AddOp { left, right })
    }
}

impl ComputationalGraph for AddOp {
    fn forward(&self) -> f64 {
        self.left.forward() + self.right.forward()
    }

    fn backward(&self, upstream_gradient: f64) {
        self.left.backward(upstream_gradient);
        self.right.backward(upstream_gradient);
    }
}
