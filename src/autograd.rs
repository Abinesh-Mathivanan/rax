use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone)]
pub struct Tensor {
    data: f64,
    grad: f64,
    grad_fn: Option<Rc<RefCell<dyn FnMut(f64)>>>,
}

impl Tensor {
    pub fn new(data: f64) -> Self {
        Tensor {
            data,
            grad: 0.0,
            grad_fn: None,
        }
    }

    pub fn data(&self) -> f64 {
        self.data
    }

    pub fn grad(&self) -> f64 {
        self.grad
    }

    pub fn set_grad(&mut self, grad: f64) {
        self.grad = grad;
    }

    pub fn set_grad_fn<F>(&mut self, grad_fn: F)
    where
        F: 'static + FnMut(f64),
    {
        self.grad_fn = Some(Rc::new(RefCell::new(grad_fn)));
    }

    pub fn backward(&mut self, grad: f64) {
        self.grad += grad;
        if let Some(ref grad_fn) = self.grad_fn {
            grad_fn.borrow_mut()(grad);
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad = 0.0;
    }
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: Self) -> Tensor {
        let mut result = Tensor::new(self.data + other.data);
        let self_clone = Rc::new(RefCell::new(self.clone()));
        let other_clone = Rc::new(RefCell::new(other.clone()));
        result.set_grad_fn(move |grad| {
            self_clone.borrow_mut().backward(grad);
            other_clone.borrow_mut().backward(grad);
        });
        result
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: Self) -> Tensor {
        let mut result = Tensor::new(self.data * other.data);
        let self_clone = Rc::new(RefCell::new(self.clone()));
        let other_clone = Rc::new(RefCell::new(other.clone()));
        let other_data = other.data;
        let self_data = self.data;
        result.set_grad_fn(move |grad| {
            self_clone.borrow_mut().backward(grad * other_data);
            other_clone.borrow_mut().backward(grad * self_data);
        });
        result
    }
}

// Optimizer
pub struct SGD {
    lr: f64,
}

impl SGD {
    pub fn new(lr: f64) -> Self {
        SGD { lr }
    }

    pub fn step(&self, params: &mut [Tensor]) {
        for param in params {
            let grad = param.grad();
            let data = param.data();
            param.set_grad(0.0);
            param.set_grad(data - self.lr * grad);
        }
    }
}

