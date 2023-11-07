use peroxide::fuga::ExpLogOps;
use revad::*;

// Example with symbol : sigmoid(x)
fn main() {
    let mut graph = Graph::default();

    let x = graph.var(1.0);
    let x_sym = Expr::Symbol(x);
    let expr_sym = 1f64 / (1f64 + (-x_sym).exp());
    println!("Expr: {:#?}", expr_sym);
    let expr = graph.parse_expr(expr_sym);

    let result = graph.forward(expr);
    println!("Result: {}", result);

    graph.backward(expr, 1.0);
    let gradient_x = graph.get_gradient(x);
    println!("Gradient x: {}", gradient_x);
}
