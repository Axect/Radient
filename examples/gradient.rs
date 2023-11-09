use radient::prelude::*;
use peroxide::fuga::{Uniform, RNG};

fn main() {
    let u = Uniform(1, 5);
    let mut l = 0usize;

    for _ in 0..100000 {
        let value = u.sample(2);
        let gradient = gradient(f, &value);
        l += gradient.len();
    }

    println!("{}", l);
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
