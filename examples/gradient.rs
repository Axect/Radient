use radient::prelude::*;
use peroxide::fuga::*;

fn main() {
    let value = vec![2.0, 1.0];
    let gradient = gradient(f, &value);
    gradient.print();
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
