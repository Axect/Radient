use revad::*;

// Example usage:
fn main() {
    let mut graph = Graph::default();

    let x_index = graph.add_var(2.0);
    let y_index = graph.add_var(3.0);

    let sum_index = graph.add_add(x_index, y_index);
    let product_index = graph.add_multiply(x_index, sum_index);

    let result = graph.forward(product_index);
    println!("Result: {}", result);

    graph.backward(product_index, 1.0);

    let gradient_x = graph.get_gradient(x_index);
    let gradient_y = graph.get_gradient(y_index);
    println!("Gradient x: {}", gradient_x);
    println!("Gradient y: {}", gradient_y);
}
