use radient::prelude::*;
use peroxide::fuga::*;

fn main() {
    let mut graph = Graph::default();
    graph.touch_vars(1);
    let symbols = graph.get_symbols();
    let expr = symbols[0].relu();
    graph.compile(expr);

    let value = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let (results, grads): (Vec<f64>, Vec<f64>) = value.iter().map(|&v| {
        let (r, g) = gradient_cached(&mut graph, &[v]);
        (r, g[0])
    }).unzip();

    println!("{:?}", value);
    println!("{:?}", results);
    println!("{:?}", grads);
}

