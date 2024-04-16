extern crate rustfft;
use rustfft::{FftPlanner, Fft};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

struct FastConvolver {
    // TODO: your fields here
}

#[derive(Debug, Clone, Copy)]
pub enum ConvolutionMode {
    TimeDomain,
    FrequencyDomain { block_size: usize },
}

impl FastConvolver {
    pub fn new(impulse_response: &[f32], mode: ConvolutionMode) -> Self {
        todo!("implement")
    }

    pub fn reset(&mut self) {
        todo!("implement")
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        todo!("implement")
    }

    pub fn flush(&mut self, output: &mut [f32]) {
        todo!("implement")
    }

    fn tdConv(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let signal_len = signal.len();
        let kernel_len = kernel.len();
        let output_len = signal_len + kernel_len - 1;
        let mut output = vec![0.0; output_len];
    
        for n in 0..output_len {
            let mut sum = 0.0;
            for k in 0..kernel_len {
                if n >= k && n - k < signal_len {
                    sum += signal[n - k] * kernel[k];
                }
            }
            output[n] = sum;
        }
    
        output
    }

    fn fdConv(signal: &mut [Complex<f32>], kernel: &[Complex<f32>]){
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft(signal.len(), rustfft::FftDirection::Forward);
    
        // Perform FFT on both signals
        fft.process(signal);
        fft.process(&mut kernel.to_vec());
    
        // Multiply the frequency representations element-wise
        for (a, b) in signal.iter_mut().zip(kernel.iter()) {
            *a *= *b;
        }
    
        // Perform inverse FFT
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft(signal.len(), rustfft::FftDirection::Inverse);
        ifft.process(signal);
    }

    // TODO: feel free to define other functions for your own use
}

// TODO: feel free to define other types (here or in other modules) for your own use
