use revad::*;

// Compute reverse mode AD
// - expression: x * (x + y)
fn main() {
    let var_x = Var::new(2.0);
    let var_y = Var::new(3.0);

    let add_op = AddOp::new(var_x.clone(), var_y.clone());
    let multiply_op = MultiplyOp::new(var_x.clone(), add_op.clone());
    println!("Multiply x: {} and y: {}", var_x.forward(), var_y.forward());

    // Forward pass to compute the result
    let result = multiply_op.forward();
    println!("Result: {}", result);

    // Backward pass to compute the gradients
    multiply_op.backward(1.0); // Assuming the gradient of the output w.r.t itself is 1

    // Accessing the gradients
    println!("Gradient of x: {}", var_x.gradient());
    println!("Gradient of y: {}", var_y.gradient());
}
