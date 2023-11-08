use radient::prelude::*;
use peroxide::fuga::*;

fn main() {
    let value = vec![2.0, 1.0];
    let gradient = gradient(f, &value);
    gradient.print();

    let mut graph = Graph::default();
    let x_vec = value.iter().map(|x| graph.var(*x)).collect::<Vec<_>>();
    let expr_vec = x_vec.iter().map(|x| Expr::Symbol(*x)).collect::<Vec<_>>();
    let expr = f(&expr_vec);
    let compiled_idx = graph.compile(expr);

    let (result, grads) = gradient_cached(&mut graph, compiled_idx, &value);
    println!("Result: {}", result);
    println!("Gradients: {:?}", grads);
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
