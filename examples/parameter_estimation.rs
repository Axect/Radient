use peroxide::fuga::*;
use radient::prelude::*;

const N: usize = 10000;
const LR: f64 = 0.001;
const EPOCHS: usize = 100;

fn main() {
    // Data Generation
    let p_true = vec![3f64, -2f64, 5f64];
    let u = Uniform(0, 10);
    let x = u.sample(N);

    let n = Normal(0.0, 0.1);
    let y = x.fmap(|t: f64| f(t, &p_true)).add_v(&n.sample(N));

    // Declare Graph
    let mut graph = Graph::default();
    graph.touch_vars(5); // p & x & y
    let p = graph.get_symbols()[0..3].to_vec();
    let x_sym = graph.get_symbols()[3].clone();
    let y_sym = graph.get_symbols()[4].clone();
    let y_hat = f(x_sym, &p);
    let loss = (y_sym - y_hat.clone()).powi(2);
    println!("loss: {:?}", loss);
    graph.compile(loss);

    // Train
    let mut loss_history = vec![0f64; EPOCHS];
    let mut wx = vec![0f64; 5];
    for li in loss_history.iter_mut() {
        let mut loss_sum = 0f64;
        for (x, y) in x.iter().zip(y.iter()) {
            wx[3] = *x;
            wx[4] = *y;
            let (loss, grad) = gradient_cached(&mut graph, &wx);
            loss_sum += loss;

            // Update weights
            for i in 0..3 {
                wx[i] -= LR * grad[i];
            }
        }
        *li = loss_sum / N as f64;
    }
    loss_history.print();
    println!("p_true: {:?}", p_true);
    println!("p_hat: {:?}", wx[0..3].to_vec());
}

pub fn f<T: Numeric<f64>>(x: T, p: &[T]) -> T {
    p[0].clone() * x.sin() + p[1].clone() * x.sqrt() + p[2].clone()
}
