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

