use radient::prelude::*;

fn main() {
    // Compile the graph
    let mut graph = Graph::default();
    graph.touch_vars(3);
    let symbols = graph.get_symbols();
    let expr = f(&symbols);
    graph.compile(expr);

    let value = vec![
        MR::ml_matrix("1 2; 3 4; 5 6"), // 3x2
        MR::ml_matrix("2; 1"),         // 2x1
        MR::ml_matrix("1; -1; 1"), // 3x1
    ];

    let (result, grads) = gradient_cached(&mut graph, &value);

    println!("result: ");
    result.print();

    println!("grads: ");
    grads.into_iter()
        .for_each(|g| { g.print(); println!() });
}

#[allow(non_snake_case)]
fn f(x_vec: &[Expr]) -> Expr {
    let A = &x_vec[0];
    let x = &x_vec[1];
    let B = &x_vec[2];

    &(A * x) + B
}
