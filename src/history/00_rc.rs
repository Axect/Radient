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

//use std::cell::RefCell;
//use std::rc::Rc;
//
//// Unique identifier for variables
//pub type VarID = usize;
//
//// The trait for the computational graph nodes
//pub trait ComputationalGraph {
//    // Computes the forward pass of the graph
//    fn forward(&self) -> f64;
//    // Performs the backward pass of the graph
//    fn backward(&self, upstream_gradient: f64);
//    // Retrieves the unique identifier of the node
//    fn id(&self) -> VarID;
//}
//
//// The Var struct represents a variable in the computational graph
//#[derive(Clone)]
//pub struct Var {
//    value: f64,
//    id: VarID,
//    // A place to store the gradient after the backward pass
//    grad: Rc<RefCell<f64>>,
//}
//
//impl Var {
//    pub fn new(value: f64, id: VarID) -> Rc<RefCell<Self>> {
//        Rc::new(RefCell::new(Var {
//            value,
//            id,
//            grad: Rc::new(RefCell::new(0.0)),
//        }))
//    }
//
//    // Function to retrieve the gradient value
//    pub fn gradient(&self) -> f64 {
//        *self.grad.borrow()
//    }
//}
//
//// Implementing the ComputationalGraph trait for Var
//impl ComputationalGraph for Var {
//    fn forward(&self) -> f64 {
//        self.value
//    }
//
//    fn backward(&self, upstream_gradient: f64) {
//        *self.grad.borrow_mut() += upstream_gradient;
//    }
//
//    fn id(&self) -> VarID {
//        self.id
//    }
//}
//
//pub struct MultiplyOp {
//    left: Rc<RefCell<dyn ComputationalGraph>>,
//    right: Rc<RefCell<dyn ComputationalGraph>>,
//}
//
//impl MultiplyOp {
//    pub fn new(
//        left: Rc<RefCell<dyn ComputationalGraph>>,
//        right: Rc<RefCell<dyn ComputationalGraph>>,
//    ) -> Rc<RefCell<Self>> {
//        Rc::new(RefCell::new(MultiplyOp { left, right }))
//    }
//}
//
//impl ComputationalGraph for MultiplyOp {
//    fn forward(&self) -> f64 {
//        self.left.borrow().forward() * self.right.borrow().forward()
//    }
//
//    fn backward(&self, upstream_gradient: f64) {
//        let left_value = self.left.borrow().forward();
//        let right_value = self.right.borrow().forward();
//
//        // Apply the chain rule: local_grad * upstream_gradient
//        let left_grad = right_value * upstream_gradient;
//        let right_grad = left_value * upstream_gradient;
//
//        // Propagate the gradients to the left and right operands
//        self.left.borrow_mut().backward(left_grad);
//        self.right.borrow_mut().backward(right_grad);
//    }
//
//    fn id(&self) -> VarID {
//        // This operation itself does not have an ID; it's not a variable but an operation.
//        panic!("MultiplyOp does not have an ID");
//    }
//}
//
//// Similarly, you would define AddOp
//pub struct AddOp {
//    left: Rc<RefCell<dyn ComputationalGraph>>,
//    right: Rc<RefCell<dyn ComputationalGraph>>,
//}
//
//impl AddOp {
//    pub fn new(
//        left: Rc<RefCell<dyn ComputationalGraph>>,
//        right: Rc<RefCell<dyn ComputationalGraph>>,
//    ) -> Rc<RefCell<Self>> {
//        Rc::new(RefCell::new(AddOp { left, right }))
//    }
//}
//
//impl ComputationalGraph for AddOp {
//    fn forward(&self) -> f64 {
//        self.left.borrow().forward() + self.right.borrow().forward()
//    }
//
//    fn backward(&self, upstream_gradient: f64) {
//        self.left.borrow_mut().backward(upstream_gradient);
//        self.right.borrow_mut().backward(upstream_gradient);
//    }
//
//    fn id(&self) -> VarID {
//        unimplemented!()
//    }
//}
//
