use rand::Rng;

pub trait Optimizer {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]);
    fn reset(&mut self);
}

pub struct SGD {
    learning_rate: f64,
}

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: usize,
}

pub struct RMSprop {
    learning_rate: f64,
    decay_rate: f64,
    epsilon: f64,
    cache: Vec<f64>,
}

pub struct AdaGrad {
    learning_rate: f64,
    epsilon: f64,
    cache: Vec<f64>,
}

pub struct Momentum {
    learning_rate: f64,
    momentum: f64,
    velocity: Vec<f64>,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        SGD { learning_rate }
    }
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl RMSprop {
    pub fn new(learning_rate: f64, decay_rate: f64, epsilon: f64) -> Self {
        RMSprop {
            learning_rate,
            decay_rate,
            epsilon,
            cache: Vec::new(),
        }
    }
}

impl AdaGrad {
    pub fn new(learning_rate: f64, epsilon: f64) -> Self {
        AdaGrad {
            learning_rate,
            epsilon,
            cache: Vec::new(),
        }
    }
}

impl Momentum {
    pub fn new(learning_rate: f64, momentum: f64) -> Self {
        Momentum {
            learning_rate,
            momentum,
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            *param -= self.learning_rate * grad;
        }
    }

    fn reset(&mut self) {}
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        if self.m.is_empty() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }

        self.t += 1;

        for ((param, grad), (m, v)) in params.iter_mut().zip(grads.iter())
            .zip(self.m.iter_mut().zip(self.v.iter_mut())) {
            *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
            *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;

            let m_hat = *m / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = *v / (1.0 - self.beta2.powi(self.t as i32));

            *param -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        if self.cache.is_empty() {
            self.cache = vec![0.0; params.len()];
        }

        for ((param, grad), cache) in params.iter_mut().zip(grads.iter())
            .zip(self.cache.iter_mut()) {
            *cache = self.decay_rate * *cache + (1.0 - self.decay_rate) * grad * grad;
            *param -= self.learning_rate * grad / (cache.sqrt() + self.epsilon);
        }
    }

    fn reset(&mut self) {
        self.cache.clear();
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        if self.cache.is_empty() {
            self.cache = vec![0.0; params.len()];
        }

        for ((param, grad), cache) in params.iter_mut().zip(grads.iter())
            .zip(self.cache.iter_mut()) {
            *cache += grad * grad;
            *param -= self.learning_rate * grad / (cache.sqrt() + self.epsilon);
        }
    }

    fn reset(&mut self) {
        self.cache.clear();
    }
}

impl Optimizer for Momentum {
    fn step(&mut self, params: &mut Vec<f64>, grads: &[f64]) {
        if self.velocity.is_empty() {
            self.velocity = vec![0.0; params.len()];
        }

        for ((param, grad), velocity) in params.iter_mut().zip(grads.iter())
            .zip(self.velocity.iter_mut()) {
            *velocity = self.momentum * *velocity - self.learning_rate * grad;
            *param += *velocity;
        }
    }

    fn reset(&mut self) {
        self.velocity.clear();
    }
}

pub struct SimpleRandomSearch {
    step_size: f64,
    rng: rand::rngs::ThreadRng,
}

pub struct GridSearch {
    step_size: f64,
    current_dim: usize,
    direction: i32,
}

impl SimpleRandomSearch {
    pub fn new(step_size: f64) -> Self {
        SimpleRandomSearch {
            step_size,
            rng: rand::thread_rng(),
        }
    }
}

impl GridSearch {
    pub fn new(step_size: f64) -> Self {
        GridSearch {
            step_size,
            current_dim: 0,
            direction: 1,
        }
    }
}

impl Optimizer for SimpleRandomSearch {
    fn step(&mut self, params: &mut Vec<f64>, _grads: &[f64]) {
        for param in params.iter_mut() {
            let perturbation = self.rng.gen_range(-self.step_size..self.step_size);
            *param += perturbation;
        }
    }

    fn reset(&mut self) {
        self.rng = rand::thread_rng();
    }
}

impl Optimizer for GridSearch {
    fn step(&mut self, params: &mut Vec<f64>, _grads: &[f64]) {
        params[self.current_dim] += self.step_size * self.direction as f64;
        self.current_dim = (self.current_dim + 1) % params.len();
        if self.current_dim == 0 {
            self.direction *= -1;
        }
    }

    fn reset(&mut self) {
        self.current_dim = 0;
        self.direction = 1;
    }
}