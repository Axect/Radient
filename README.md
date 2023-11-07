# Radient

Radient is a Rust library designed for automatic differentiation. It leverages the power of computational graphs to perform forward and backward passes for gradient calculations.

## Features

- Implementation of computational graphs.
- Forward and backward propagation for gradient computation.
- Support for various operations like exponential, logarithmic, power, and trigonometric functions.

## Examples

### Example 1: Basic Operations with Symbols

```rust
use peroxide::fuga::{ExpLogOps, PowOps, TrigOps};
use radient::*;

// Example with symbol: ln(x + y) * tanh(x - y)^2
fn main() {
    let mut graph = Graph::default();

    let x = graph.var(2.0);
    let y = graph.var(1.0);
    let x_sym = Expr::Symbol(x);
    let y_sym = Expr::Symbol(y);
    let expr_sym = (&x_sym + &y_sym).ln() * (&x_sym - &y_sym).tanh().powi(2);

    let expr = graph.parse_expr(expr_sym);

    let result = graph.forward(expr);
    println!("Result: {}", result);

    graph.backward(expr, 1.0);
    let gradient_x = graph.get_gradient(x);
    println!("Gradient x: {}", gradient_x);
}
```

## Getting Started

To use Radient in your project, add the following to your `Cargo.toml`:

```toml
[dependencies]
radient = "0.1"
```

Then, add the following code in your Rust file:

```rust
use radient::*;
```

## License

Radient is licensed under the Apache2.0 or MIT license - see the [LICENSE-Apache2.0](./LICENSE-Apache2.0) & [LICENSE-MIT](./LICENSE-MIT) file for details.

