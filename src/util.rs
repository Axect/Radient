use crate::core::{Graph, Expr};

pub fn gradient<F: Fn(&[Expr]) -> Expr>(f: F, x: &[f64]) -> Vec<f64> {
    let mut graph = Graph::default();
    let var_vec = x.iter().map(|x| graph.var(*x)).collect::<Vec<_>>();
    let expr_vec = var_vec.iter().map(|x| Expr::Symbol(*x)).collect::<Vec<_>>();
    let result_expr = f(&expr_vec);

    let compiled = graph.compile(result_expr);
    let _ = graph.forward(compiled);
    graph.backward(compiled, 1.0);

    var_vec.iter().map(|x| graph.get_gradient(*x)).collect::<Vec<_>>()
}

/// graph is already compiled
pub fn gradient_cached(g: &mut Graph, compiled_idx: usize, x: &[f64]) -> (f64, Vec<f64>) {
    g.reset();
    g.subs_vars(x);
    let result = g.forward(compiled_idx);
    g.backward(compiled_idx, 1.0);
    let vars = g.get_vars();
    let grads = vars.iter().map(|x| g.get_gradient(*x)).collect::<Vec<_>>();

    (result, grads)
}
