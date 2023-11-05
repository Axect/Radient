extern crate typed_arena;

use typed_arena::Arena;

#[derive(Default)]
pub struct Graph {
    arena: Arena<Node>,
    gradients: Vec<f64>,
    value_buffer: Vec<f64>,
    nodes: Vec<Node>, // Added to store the nodes
}

enum Node {
    Var(usize),             // Index in the value buffer
    Add(usize, usize),      // Indices of the left and right operands
    Multiply(usize, usize), // Indices of the left and right operands
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

    pub fn forward(&self, index: usize) -> f64 {
        match self.nodes[index] {
            Node::Var(value_index) => self.value_buffer[value_index],
            Node::Add(left_index, right_index) => {
                self.forward(left_index) + self.forward(right_index)
            }
            Node::Multiply(left_index, right_index) => {
                self.forward(left_index) * self.forward(right_index)
            }
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
        }
    }

    pub fn get_gradient(&self, var_index: usize) -> f64 {
        self.gradients[var_index]
    }
}
