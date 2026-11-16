//! Small deterministic MLP used only to nominate specification comparisons.

use crate::MboostrsError;
use ndarray::ArrayView2;
use rand::{seq::SliceRandom, RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};

const WIDTH: usize = 32;
const MAX_ROWS: usize = 2000;
const MAX_INPUTS: usize = 256;

#[derive(Clone)]
struct Network {
    inputs: usize,
    parameters: Vec<f64>,
}

impl Network {
    fn offsets(&self) -> (usize, usize, usize, usize, usize) {
        let b1 = self.inputs * WIDTH;
        let w2 = b1 + WIDTH;
        let b2 = w2 + WIDTH * WIDTH;
        let out = b2 + WIDTH;
        (b1, w2, b2, out, out + WIDTH)
    }

    fn new(inputs: usize, rng: &mut ChaCha8Rng) -> Self {
        let mut network = Self {
            inputs,
            parameters: vec![0.0; inputs * WIDTH + WIDTH * WIDTH + 3 * WIDTH + 1],
        };
        let (b1, w2, b2, out, bias) = network.offsets();
        for (start, end, fan_in, fan_out) in [
            (0, b1, inputs, WIDTH),
            (w2, b2, WIDTH, WIDTH),
            (out, bias, WIDTH, 1),
        ] {
            let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();
            for parameter in &mut network.parameters[start..end] {
                *parameter = rng.random_range(-scale..scale);
            }
        }
        network
    }

    fn forward(&self, x: &[f64], a: &mut [f64; WIDTH], b: &mut [f64; WIDTH]) -> f64 {
        let (b1, w2, b2, out, bias) = self.offsets();
        for j in 0..WIDTH {
            a[j] = (self.parameters[b1 + j]
                + (0..self.inputs)
                    .map(|i| x[i] * self.parameters[i * WIDTH + j])
                    .sum::<f64>())
            .tanh();
        }
        for j in 0..WIDTH {
            b[j] = (self.parameters[b2 + j]
                + (0..WIDTH)
                    .map(|i| a[i] * self.parameters[w2 + i * WIDTH + j])
                    .sum::<f64>())
            .tanh();
        }
        self.parameters[bias]
            + (0..WIDTH)
                .map(|j| b[j] * self.parameters[out + j])
                .sum::<f64>()
    }

    /// Gradient of half squared error, accumulated into a minibatch buffer.
    fn gradient(&self, x: &[f64], y: f64, grad: &mut [f64]) {
        let (b1, w2, b2, out, bias) = self.offsets();
        let (mut a, mut b) = ([0.0; WIDTH], [0.0; WIDTH]);
        let error = self.forward(x, &mut a, &mut b) - y;
        let mut delta = [0.0; WIDTH];
        grad[bias] += error;
        for j in 0..WIDTH {
            grad[out + j] += error * b[j];
            delta[j] = error * self.parameters[out + j] * (1.0 - b[j] * b[j]);
            grad[b2 + j] += delta[j];
        }
        for i in 0..WIDTH {
            let mut back = 0.0;
            for j in 0..WIDTH {
                grad[w2 + i * WIDTH + j] += a[i] * delta[j];
                back += self.parameters[w2 + i * WIDTH + j] * delta[j];
            }
            back *= 1.0 - a[i] * a[i];
            grad[b1 + i] += back;
            for k in 0..self.inputs {
                grad[k * WIDTH + i] += x[k] * back;
            }
        }
    }
}

pub(super) struct Neural {
    network: Network,
    categorical: Vec<usize>,
    means: Vec<f64>,
    scales: Vec<f64>,
    pub metadata: Value,
}

impl Neural {
    fn encode(&self, row: impl Iterator<Item = f64>, encoded: &mut [f64]) {
        encoded.fill(0.0);
        let mut offset = 0;
        for (j, value) in row.enumerate() {
            if self.categorical[j] == 0 {
                encoded[offset] = ((value - self.means[j]) / self.scales[j]).clamp(-5.0, 5.0);
                offset += 1;
            } else {
                encoded[offset + value as usize] = 1.0;
                offset += self.categorical[j];
            }
        }
    }

    pub fn predict(&self, x: ArrayView2<f64>) -> Vec<f64> {
        let mut encoded = vec![0.0; self.network.inputs];
        let (mut a, mut b) = ([0.0; WIDTH], [0.0; WIDTH]);
        x.rows()
            .into_iter()
            .map(|row| {
                self.encode(row.iter().copied(), &mut encoded);
                self.network.forward(&encoded, &mut a, &mut b)
            })
            .collect()
    }

    pub fn fit(
        x: ArrayView2<f64>,
        y: &[f64],
        categorical: &[usize],
        seed: u64,
        rounds: u32,
    ) -> Result<Self, MboostrsError> {
        let inputs = categorical
            .iter()
            .try_fold(0usize, |sum, count| sum.checked_add((*count).max(1)))
            .ok_or_else(|| MboostrsError::Control("neural input width overflows".into()))?;
        if inputs > MAX_INPUTS {
            return Err(MboostrsError::Control(format!(
                "neural pre-scan supports at most {MAX_INPUTS} encoded inputs, got {inputs}"
            )));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..x.nrows()).collect();
        indices.shuffle(&mut rng);
        indices.truncate(MAX_ROWS);
        let n_valid = (indices.len() / 5).max(1);
        let (validation, training) = indices.split_at(n_valid);
        let mut model = Self {
            network: Network::new(inputs, &mut rng),
            categorical: categorical.to_vec(),
            means: vec![0.0; x.ncols()],
            scales: vec![1.0; x.ncols()],
            metadata: Value::Null,
        };
        for j in 0..x.ncols() {
            if categorical[j] == 0 {
                let mean = training.iter().map(|i| x[[*i, j]]).sum::<f64>() / training.len() as f64;
                let scale = (training
                    .iter()
                    .map(|i| (x[[*i, j]] - mean).powi(2))
                    .sum::<f64>()
                    / training.len() as f64)
                    .sqrt();
                model.means[j] = mean;
                model.scales[j] = if scale > 1e-12 { scale } else { 1.0 };
            }
        }
        let encoded: Vec<Vec<f64>> = indices
            .iter()
            .map(|i| {
                let mut row = vec![0.0; inputs];
                model.encode(x.row(*i).iter().copied(), &mut row);
                row
            })
            .collect();
        let loss = |network: &Network| {
            let (mut a, mut b) = ([0.0; WIDTH], [0.0; WIDTH]);
            (0..n_valid)
                .map(|i| (network.forward(&encoded[i], &mut a, &mut b) - y[validation[i]]).powi(2))
                .sum::<f64>()
                / n_valid as f64
        };
        let mut best = model.network.clone();
        let mut best_loss = loss(&best);
        let (mut best_epoch, mut epochs) = (0, 0);
        let mut first = vec![0.0; best.parameters.len()];
        let mut second = first.clone();
        let mut grad = first.clone();
        let (mut beta1, mut beta2) = (1.0, 1.0);
        let mut order: Vec<usize> = (n_valid..indices.len()).collect();
        for epoch in 1..=rounds.min(600) {
            epochs = epoch;
            order.shuffle(&mut rng);
            for batch in order.chunks(64) {
                grad.fill(0.0);
                for &i in batch {
                    model
                        .network
                        .gradient(&encoded[i], y[indices[i]], &mut grad);
                }
                for (g, parameter) in grad.iter_mut().zip(&model.network.parameters) {
                    *g = *g / batch.len() as f64 + 1e-4 * parameter;
                }
                let norm = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                let clip = if norm > 5.0 { 5.0 / norm } else { 1.0 };
                beta1 *= 0.9;
                beta2 *= 0.999;
                for i in 0..grad.len() {
                    let g = grad[i] * clip;
                    first[i] = 0.9 * first[i] + 0.1 * g;
                    second[i] = 0.999 * second[i] + 0.001 * g * g;
                    model.network.parameters[i] -= 0.003 * (first[i] / (1.0 - beta1))
                        / ((second[i] / (1.0 - beta2)).sqrt() + 1e-8);
                }
            }
            if epoch % 10 == 0 || epoch == rounds.min(600) {
                let current = loss(&model.network);
                if !current.is_finite() {
                    return Err(MboostrsError::Solver(
                        "neural pre-scan loss is nonfinite".into(),
                    ));
                }
                if current < best_loss {
                    best_loss = current;
                    best_epoch = epoch;
                    best = model.network.clone();
                }
                if epoch >= best_epoch + 60 {
                    break;
                }
            }
        }
        model.network = best;
        model.metadata = json!({"hidden_widths":[WIDTH,WIDTH],"activation":"tanh",
            "optimizer":"adam","learning_rate":0.003,"encoded_inputs":inputs,
            "fit_rows":training.len(),"validation_rows":n_valid,"epochs":epochs,
            "best_epoch":best_epoch,"validation_mse_standardized":best_loss,
            "numeric_clip":5.0,"validation_scope":"internal screening fit; not adoption evidence"});
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backprop_matches_finite_differences() {
        let network = Network::new(3, &mut ChaCha8Rng::seed_from_u64(6));
        let x = [0.2, -0.7, 1.1];
        let mut gradient = vec![0.0; network.parameters.len()];
        network.gradient(&x, 0.4, &mut gradient);
        for i in (0..gradient.len()).step_by(19) {
            let mut perturbed = network.clone();
            let evaluate = |n: &Network| {
                let (mut a, mut b) = ([0.0; WIDTH], [0.0; WIDTH]);
                0.5 * (n.forward(&x, &mut a, &mut b) - 0.4).powi(2)
            };
            perturbed.parameters[i] += 1e-6;
            let plus = evaluate(&perturbed);
            perturbed.parameters[i] -= 2e-6;
            let minus = evaluate(&perturbed);
            assert!(
                (gradient[i] - (plus - minus) / 2e-6).abs() < 1e-7,
                "parameter {i}"
            );
        }
    }
}
