use radient::prelude::*;

fn main() {
    let value = vec![2f64, 1f64];
    let (result, gradient) = gradient(f, &value);
    println!("result: {}, gradient: {:?}", result, gradient);
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
