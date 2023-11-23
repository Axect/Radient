# Radient

Radient is a Rust library designed for automatic differentiation. It leverages the power of computational graphs to perform forward and backward passes for gradient calculations.

## Features

- Implementation of computational graphs.
- Forward and backward propagation for gradient computation.
- Support for various operations like exponential, logarithmic, power, and trigonometric functions.

## Examples

### Example 1: Basic Operations with Symbols

```rust
use radient::prelude::*;

// Example with symbol : ln(x + y) * tanh(x - y)^2
fn main() {
    let mut graph = Graph::default();

    let x = graph.var(2.0);
    let y = graph.var(1.0);
    let x_sym = Expr::Symbol(x);
    let y_sym = Expr::Symbol(y);
    let expr_sym = (&x_sym + &y_sym).ln() * (&x_sym - &y_sym).tanh().powi(2);

    graph.compile(expr_sym);

    let result = graph.forward();
    println!("Result: {}", result);

    graph.backward();
    let gradient_x = graph.get_gradient(x);
    println!("Gradient x: {}", gradient_x);
}
```

### Example 2: Obtain gradient of a function

For gradient, you have two options:

1. `gradient`: Concise but relatively slow (but not too much)
2. `gradient_cached`: Fast but little bit verbose

#### 2.1: `gradient`

```rust
use radient::prelude::*;

fn main() {
    let value = vec![2f64, 1f64];
    // No cached gradient - concise but relatively slow
    let (result, gradient) = gradient(f, &value);
    println!("result: {}, gradient: {:?}", result, gradient);
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
```

#### 2.2: `gradient_cached`

```rust
use radient::prelude::*;

fn main() {
    // Compile the graph
    let mut graph = Graph::default();
    graph.touch_vars(2);
    let symbols = graph.get_symbols();
    let expr = f(&symbols);
    graph.compile(expr);

    // Compute
    let value = vec![2f64, 1f64];
    let (result, grads) = gradient_cached(&mut graph, &value);

    println!("result: {}, gradient: {:?}", result, grads);
}

fn f(x_vec: &[Expr]) -> Expr {
    let x = &x_vec[0];
    let y = &x_vec[1];

    (x.powi(2) + y.powi(2)).sqrt()
}
```

### Example 3: Single layer perceptron (low-level)

```rust
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

    // Group 1 (Normal(2, 0.5), Normal(1, 0.5))
    let n1_x1 = Normal(2.0, 0.5);
    let n1_x2 = Normal(1.0, 0.5);

    // Group 2 (Normal(-3, 0.5), Normal(-2, 0.5))
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
    graph.touch_vars(7); // w & x & y = 3 + 3 + 1
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
        }
    }
    let total = 2 * n;
    println!("Accuracy: {}%", correct as f64 / total as f64 * 100f64);
    println!("Weights: {:?}", wx);
}
```

## Getting Started

To use Radient in your project, add the following to your `Cargo.toml`:

```toml
[dependencies]
radient = "0.2"
```

Then, add the following code in your Rust file:

```rust
use radient::*;
```

## License

Radient is licensed under the Apache2.0 or MIT license - see the [LICENSE-APACHE](./LICENSE-APACHE) & [LICENSE-MIT](./LICENSE-MIT) file for details.

