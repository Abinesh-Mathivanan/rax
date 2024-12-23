#[cfg(test)]
mod tests {
    use ndarray::array;
    use rax::tensor::{dot, determinant};

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
    

    #[test]
    fn test_dot() {
        let input1 = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let input2 = array![[5.0, 6.0], [7.0, 8.0]].into_dyn();
        let result = dot(&input1, &input2);
        let expected = array![[19.0, 22.0], [43.0, 50.0]].into_dyn();
        assert!(result.abs_diff_eq(&expected, 1e-6));
    }

    #[test]
    fn test_determinant() {
        let input = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let result = determinant(&input);
        let expected = -2.0;
        assert!((result - expected).abs() < 1e-6);
    }
}