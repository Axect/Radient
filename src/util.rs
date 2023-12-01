use crate::core::{Expr, Graph};
use peroxide_num::Numeric;
use std::ops::Div;
use crate::traits::{ActivationFunction, Matrizable};

pub fn gradient<F: Fn(&[Expr]) -> Expr>(f: F, x: &[f64]) -> (f64, Vec<f64>) {
    let mut graph = Graph::default();
    let var_vec = x.iter().map(|x| graph.var(*x)).collect::<Vec<_>>();
    let expr_vec = var_vec.iter().map(|x| Expr::Symbol(*x)).collect::<Vec<_>>();
    let result_expr = f(&expr_vec);

    graph.compile(result_expr);
    let result = graph.forward();
    graph.backward();

    let grads = var_vec
        .iter()
        .map(|x| graph.get_gradient(*x))
        .collect::<Vec<_>>();

    (result, grads)
}

/// graph is already compiled
pub fn gradient_cached<T: std::fmt::Debug + Numeric<f64> + Default + ActivationFunction + Matrizable>(
    g: &mut Graph<T>,
    x: &[T],
) -> (T, Vec<T>)
where
    f64: Div<T, Output = T>,
{
    g.reset();
    g.subs_vars(x);
    let result = g.forward();
    //println!("result: {:?}", result);
    g.backward();
    let grads = g.get_gradients();

    (result, grads)
}
