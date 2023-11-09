use radient::prelude::*;

fn main() {
    // Compile the graph
    let mut graph = Graph::default();
    graph.touch_vars(2);
    let symbols = graph.get_symbols();
    let expr = f(&symbols);
    graph.compile(expr);

    let value = vec![2f64, 1f64];
    let (result, grads) = gradient_cached(&mut graph, &value);

    println!("result: {}, gradient: {:?}", result, grads);
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
