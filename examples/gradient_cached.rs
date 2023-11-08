use radient::prelude::*;
use peroxide::fuga::*;

fn main() {
    let u = Uniform(1, 5);
    let mut l = 0usize;

    let mut graph = Graph::default();
    graph.touch_vars(2);
    let symbols = graph.get_symbols();
    let expr = f(&symbols);
    graph.compile(expr);

    for _ in 0 .. 100000 {
        let value = u.sample(2);
        let (_, grads) = gradient_cached(&mut graph, &value);
        l += grads.len();
    }

    println!("{}", l);
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
