use peroxide::fuga::*;
use radient::prelude::*;

// Single Layer Perceptron to solve the classification problem
//
// y = sigmoid(sum(w * x))
//
// - x : 1, input (1+2-dim vector)
// - y : label (0 or 1)
// - w : weight (3-dim vector)
fn main() {
    // Data Generation
    let n = 100;

    let n1_x1 = Normal(2.0, 0.5);
    let n1_x2 = Normal(1.0, 0.5);

    let n2_x1 = Normal(-3.0, 0.5);
    let n2_x2 = Normal(-2.0, 0.5);

    let group1: Vec<_> = n1_x1
        .sample(n)
        .into_iter()
        .zip(n1_x2.sample(n))
        .map(|(x1, x2)| vec![1.0, x1, x2])
        .collect();
    let group2: Vec<_> = n2_x1
        .sample(n)
        .into_iter()
        .zip(n2_x2.sample(n))
        .map(|(x1, x2)| vec![1.0, x1, x2])
        .collect();
    let label1 = vec![0f64; n];
    let label2 = vec![1f64; n];

    let data: Vec<_> = group1.into_iter().chain(group2).collect();
    let labels: Vec<_> = label1.into_iter().chain(label2).collect();

    // Declare Graph
    let mut graph = Graph::default();
    graph.touch_vars(7); // w & x & y
    let w = graph.get_symbols()[0..3].to_vec();
    let x = graph.get_symbols()[3..6].to_vec();
    let y = graph.get_symbols()[6].clone();
    let expr: Expr = w.into_iter().zip(x).map(|(w, x)| w * x).sum();
    let y_hat = expr.sigmoid();
    let loss = (y - y_hat.clone()).powi(2);
    println!("loss: {:?}", loss);
    graph.compile(loss);

    // Train
    let lr = 0.1;
    let epochs = 100;
    let mut loss_history = vec![0f64; epochs];

    let mut wx = vec![0f64; 7];
    for li in loss_history.iter_mut() {
        let mut loss_sum = 0f64;
        for (x, y) in data.iter().zip(labels.iter()) {
            wx[3..6].copy_from_slice(x);
            wx[6] = *y;
            let (loss, grad) = gradient_cached(&mut graph, &wx);
            loss_sum += loss;

            // Update weights
            for i in 0..3 {
                wx[i] -= lr * grad[i];
            }
        }
        *li = loss_sum / n as f64;
    }

    loss_history.print();

    // Test
    let mut correct = 0;
    let mut wrong = 0;
    graph.compile(y_hat);

    for (x, y) in data.iter().zip(labels) {
        wx[3..6].copy_from_slice(x);
        wx[6] = y;
        graph.reset();
        graph.subs_vars(&wx);
        let y_hat = graph.forward();
        let c_hat = if y_hat > 0.5 { 1.0 } else { 0.0 };
        if c_hat == y {
            correct += 1;
        } else {
            wrong += 1;
        }
    }

    let total = correct + wrong;
    println!("correct: {}, wrong: {}, total: {}", correct, wrong, total);
    println!("Accuracy: {}%", correct as f64 / total as f64 * 100f64);
    println!("Weights: {:?}", wx);
}
