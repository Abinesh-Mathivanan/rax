use ndarray::{Array1, Array, Axis, IxDyn};
use ndarray_linalg::solve::Determinant;


/// Computes the softmax of a 1D array.
pub fn softmax(input: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let input_1d = input.view().into_dimensionality::<ndarray::Ix1>().unwrap(); // Use view to avoid cloning
    let max = input_1d.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Numerical stability
    let exp_values: Array1<f64> = input_1d.mapv(|x| (x - max).exp());
    let sum = exp_values.sum();
    (exp_values / sum).into_dyn()
}

/// Computes the softmax along a specific axis of a 2D array.
pub fn softmax_2d(input: &Array<f64, IxDyn>, axis: Axis) -> Array<f64, IxDyn> {
    let input_2d = input.view().into_dimensionality::<ndarray::Ix2>().unwrap(); // Use view to avoid cloning
    let mut output = input_2d.to_owned();
    output.map_axis_mut(axis, |mut row| {
        let row_owned = row.to_owned(); // Convert to owned array
        row.assign(&softmax(&row_owned.into_dyn()));
    });
    output.into_dyn()
}

/// Computes the log-sum-exp of a 1D array.
pub fn logsumexp(input: &Array<f64, IxDyn>) -> f64 {
    let input_1d = input.view().into_dimensionality::<ndarray::Ix1>().unwrap(); // Use view to avoid cloning
    let max = input_1d.iter().cloned().fold(f64::NEG_INFINITY, f64::max); // Numerical stability
    let sum_exp = input_1d.mapv(|x| (x - max).exp()).sum();
    max + sum_exp.ln()
}

/// Computes the log-sum-exp along a specific axis of a 2D array.
pub fn logsumexp_2d(input: &Array<f64, IxDyn>, axis: Axis) -> Array1<f64> {
    let input_2d = input.view().into_dimensionality::<ndarray::Ix2>().unwrap(); // Use view to avoid cloning
    input_2d.map_axis(axis, |row| logsumexp(&row.to_owned().into_dyn()))
}

/// Normalizes a 1D array to have a range of [0, 1].
pub fn normalize_minmax(input: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let input_1d = input.view().into_dimensionality::<ndarray::Ix1>().unwrap(); // Use view to avoid cloning
    let min = input_1d.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = input_1d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    input_1d.mapv(|x| (x - min) / (max - min)).into_dyn()
}

/// Normalizes a 1D array to have zero mean and unit variance.
pub fn normalize_zscore(input: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let input_1d = input.view().into_dimensionality::<ndarray::Ix1>().unwrap(); // Use view to avoid cloning
    let mean = input_1d.mean().unwrap_or(0.0);
    let std = input_1d.std(0.0);
    input_1d.mapv(|x| (x - mean) / std).into_dyn()
}

/// Sums all elements in the tensor.
pub fn sum_all(input: &Array<f64, IxDyn>) -> f64 {
    input.sum()
}

/// Sums elements along a specific axis.
pub fn sum_axis(input: &Array<f64, IxDyn>, axis: Axis) -> Array<f64, IxDyn> {
    input.sum_axis(axis).into_dyn()
}

/// Computes the mean of all elements in the tensor.
pub fn mean_all(input: &Array<f64, IxDyn>) -> f64 {
    input.mean().unwrap_or(0.0)
}

/// Computes the mean along a specific axis.
pub fn mean_axis(input: &Array<f64, IxDyn>, axis: Axis) -> Array<f64, IxDyn> {
    input.mean_axis(axis).unwrap().into_dyn()
}

/// Finds the maximum value along an axis.
pub fn max_axis(input: &Array<f64, IxDyn>, axis: Axis) -> Array<f64, IxDyn> {
    input.map_axis(axis, |view| *view.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()).into_dyn()
}

/// Finds the minimum value along an axis.
pub fn min_axis(input: &Array<f64, IxDyn>, axis: Axis) -> Array<f64, IxDyn> {
    input.map_axis(axis, |view| *view.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()).into_dyn()
}

/// Reshapes the input tensor to the specified shape.
pub fn reshape(input: Array<f64, IxDyn>, new_shape: &[usize]) -> Array<f64, IxDyn> {
    input.into_shape(new_shape).unwrap()
}

/// Transposes the input tensor, swapping its axes.
pub fn transpose(input: Array<f64, IxDyn>, axes: Option<&[usize]>) -> Array<f64, IxDyn> {
    if let Some(axes) = axes {
        input.permuted_axes(axes).to_owned()
    } else {
        input.t().to_owned()
    }
}

/// Broadcasts the input tensor to a new shape.
pub fn broadcast(input: &Array<f64, IxDyn>, new_shape: &[usize]) -> Array<f64, IxDyn> {
    input.broadcast(new_shape).unwrap().to_owned()
}

pub fn dot(input1: &Array<f64, IxDyn>, input2: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    let matrix1 = input1.to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    let matrix2 = input2.to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    matrix1.dot(&matrix2).into_dyn()
}

pub fn determinant(input: &Array<f64, IxDyn>) -> f64 {
    let matrix = input.to_owned().into_dimensionality::<ndarray::Ix2>().unwrap();
    matrix.det().unwrap()
}
