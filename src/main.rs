use peroxide::fuga::{ExpLogOps, PowOps, TrigOps};
use revad::*;

// Example with symbol : sigmoid(x)
fn main() {
    let mut graph = Graph::default();

    let x = graph.var(1.0);
    let y = graph.var(2.0);
    let x_sym = Expr::Symbol(x);
    let y_sym = Expr::Symbol(y);
    let z_sym = (&x_sym + &y_sym).sin() / (&x_sym + &y_sym).cos();
    let mut expr_sym = &z_sym.sin() + &z_sym.cos();
    for _ in 0..12 {
        expr_sym = &expr_sym.sin() + &expr_sym.cos();
    }
    let expr = graph.parse_expr(expr_sym);

    let result = graph.forward(expr);
    println!("Result: {}", result);

    graph.backward(expr, 1.0);
    let gradient_x = graph.get_gradient(x);
    println!("Gradient x: {}", gradient_x);
}
