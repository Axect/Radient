use peroxide::fuga::{PowOps, TrigOps};
use revad::*;
use std::f64::consts::PI;

// Example with symbol : sin(x)^2 * y
fn main() {
    let mut graph = Graph::default();

    let x = graph.var(PI / 2f64);
    let y = graph.var(2.0);

    let x_sym = Expr::Symbol(x);
    let y_sym = Expr::Symbol(y);

    let expr_sym = x_sym.sin().powi(2) * y_sym;
    let expr = parse_expr(expr_sym, &mut graph);
    let result = graph.forward(expr);
    println!("Result: {}", result);

    graph.backward(expr, 1.0);
    let gradient_x = graph.get_gradient(x);
    let gradient_y = graph.get_gradient(y);
    println!("Gradient x: {}", gradient_x);
    println!("Gradient y: {}", gradient_y);
}

//// Example usage: sin(x)^2 * y
//fn main() {
//    let mut graph = Graph::default();
//
//    let x = graph.add_var(PI / 2f64);
//    let y = graph.add_var(2.0);
//
//    let expr_1 = graph.add_sin(x);
//    let expr_2 = graph.add_mul(expr_1, expr_1);
//    let expr = graph.add_mul(expr_2, y);
//
//    let result = graph.forward(expr);
//    println!("Result: {}", result);
//
//    graph.backward(expr, 1.0);
//
//    let gradient_x = graph.get_gradient(x);
//    let gradient_y = graph.get_gradient(y);
//    println!("Gradient x: {}", gradient_x);
//    println!("Gradient y: {}", gradient_y);
//}
