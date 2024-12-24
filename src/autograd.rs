use ndarray::Array;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

#[derive(Clone)]
pub struct Tensor {
    pub data: Array<f64, ndarray::IxDyn>,
    pub grad: Option<Array<f64, ndarray::IxDyn>>,
    pub requires_grad: bool,
    pub creator: Option<Weak<RefCell<GraphNode>>>,
}

impl Tensor {
    pub fn new(data: Array<f64, ndarray::IxDyn>, requires_grad: bool) -> Self {
        Tensor {
            data,
            grad: None,
            requires_grad,
            creator: None,
        }
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn backward(&mut self) {
        if self.grad.is_none() {
            self.grad = Some(Array::ones(self.data.raw_dim()));
        }

        let mut stack = vec![Rc::new(RefCell::new(self.clone()))];

        while let Some(node) = stack.pop() {
            if let Some(creator_weak) = &node.borrow().creator {
                if let Some(creator) = creator_weak.upgrade() {
                    let grad = node.borrow().grad.clone().unwrap();

                    {
                        let backward_fn = &creator.borrow().backward_fn;
                        backward_fn(&grad, &mut creator.borrow_mut().inputs);
                    }

                    for input in &creator.borrow().inputs {
                        stack.push(input.clone());
                    }
                }
            }
        }
    }
}

pub struct GraphNode {
    pub operation: String,
    pub inputs: Vec<Rc<RefCell<Tensor>>>,
    pub backward_fn: Box<dyn Fn(&Array<f64, ndarray::IxDyn>, &mut Vec<Rc<RefCell<Tensor>>>)>,
}

impl GraphNode {
    pub fn new(
        operation: String,
        inputs: Vec<Rc<RefCell<Tensor>>>,
        backward_fn: Box<dyn Fn(&Array<f64, ndarray::IxDyn>, &mut Vec<Rc<RefCell<Tensor>>>)>,
    ) -> Self {
        GraphNode {
            operation,
            inputs,
            backward_fn,
        }
    }
}

pub fn add(tensor1: &Rc<RefCell<Tensor>>, tensor2: &Rc<RefCell<Tensor>>) -> Rc<RefCell<Tensor>> {
    let data = &tensor1.borrow().data + &tensor2.borrow().data;
    let requires_grad = tensor1.borrow().requires_grad || tensor2.borrow().requires_grad;

    let output = Rc::new(RefCell::new(Tensor::new(data, requires_grad)));

    if requires_grad {
        let node = GraphNode::new(
            "add".to_string(),
            vec![tensor1.clone(), tensor2.clone()],
            Box::new(move |grad, inputs| {
                inputs[0].borrow_mut().grad = Some(grad.clone());
                inputs[1].borrow_mut().grad = Some(grad.clone());
            }),
        );
        output.borrow_mut().creator = Some(Rc::downgrade(&Rc::new(RefCell::new(node))));
    }

    output
}
