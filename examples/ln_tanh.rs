use radient::prelude::*;

// Example with symbol : ln(x + y) * tanh(x - y)^2
fn main() {
    let mut graph = Graph::default();

    let x = graph.var(2.0);
    let y = graph.var(1.0);
    let x_sym = Expr::Symbol(x);
    let y_sym = Expr::Symbol(y);
    let expr_sym = (&x_sym + &y_sym).ln() * (&x_sym - &y_sym).tanh().powi(2);

    graph.compile(expr_sym);

    let result = graph.forward();
    println!("Result: {}", result);

    graph.backward();
    let gradient_x = graph.get_gradient(x);
    println!("Gradient x: {}", gradient_x);
}
