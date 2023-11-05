#[derive(Default)]
pub struct Graph {
    gradients: Vec<f64>,
    value_buffer: Vec<f64>,
    nodes: Vec<Node>, // Added to store the nodes
}

enum Node {
    Var(usize),             // Index in the value buffer
    Add(usize, usize),      // Indices of the left and right operands
    Multiply(usize, usize), // Indices of the left and right operands
    Sin(usize),             // Index of the operand
}

impl Graph {
    pub fn add_var(&mut self, value: f64) -> usize {
        let index = self.value_buffer.len();
        self.value_buffer.push(value);
        self.gradients.push(0.0);
        self.nodes.push(Node::Var(index));
        index // The index is used to refer to this variable
    }

    pub fn add_add(&mut self, left: usize, right: usize) -> usize {
        let index = self.nodes.len();
        self.gradients.push(0.0);
        self.nodes.push(Node::Add(left, right));
        index
    }

    pub fn add_multiply(&mut self, left: usize, right: usize) -> usize {
        let index = self.nodes.len();
        self.gradients.push(0.0);
        self.nodes.push(Node::Multiply(left, right));
        index
    }

    pub fn add_sin(&mut self, operand: usize) -> usize {
        let index = self.nodes.len();
        self.gradients.push(0.0);
        self.nodes.push(Node::Sin(operand));
        index
    }

    pub fn forward(&self, index: usize) -> f64 {
        match self.nodes[index] {
            Node::Var(value_index) => self.value_buffer[value_index],
            Node::Add(left_index, right_index) => {
                self.forward(left_index) + self.forward(right_index)
            }
            Node::Multiply(left_index, right_index) => {
                self.forward(left_index) * self.forward(right_index)
            }
            Node::Sin(operand_index) => self.forward(operand_index).sin(),
        }
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
            Node::Multiply(left_index, right_index) => {
                let left_val = self.forward(left_index);
                let right_val = self.forward(right_index);
                self.backward(left_index, right_val * upstream_gradient);
                self.backward(right_index, left_val * upstream_gradient);
            }
            Node::Sin(operand_index) => {
                let operand_val = self.forward(operand_index);
                self.backward(operand_index, operand_val.cos() * upstream_gradient);
            }
        }
    }

    pub fn get_gradient(&self, var_index: usize) -> f64 {
        self.gradients[var_index]
    }

    // Method to create a multiplication operation
    pub fn mul(&mut self, left: usize, right: usize) -> usize {
        self.add_multiply(left, right)
    }

    // Method to create an addition operation
    pub fn add(&mut self, left: usize, right: usize) -> usize {
        self.add_add(left, right)
    }
}

// Define a macro to simplify expression building
#[macro_export]
macro_rules! expr {
    // Create a new variable node from a literal value
    ($graph:expr; val $value:expr) => {
        $graph.add_var($value)
    };
    // Refer to an existing variable node by index
    ($graph:expr; var $index:expr) => {
        $index
    };
    // Create an add node from two sub-expressions
    ($graph:expr; add $left:tt, $right:tt) => {
        $graph.add_add(expr!($graph; $left), expr!($graph; $right))
    };
    // Create a multiply node from two sub-expressions
    ($graph:expr; mul $left:tt, $right:tt) => {
        $graph.add_multiply(expr!($graph; $left), expr!($graph; $right))
    };
    // Add more operations as needed
}
