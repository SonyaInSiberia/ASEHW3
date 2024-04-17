extern crate rustfft;
use rustfft::{FftPlanner, Fft};
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use realfft::RealToComplex;
use realfft::RealFftPlanner;

struct FastConvolver {
    // TODO: your fields here
    impulse_response: Vec<f32>,
    mode: ConvolutionMode,

}

#[derive(Debug, Clone, Copy)]
pub enum ConvolutionMode {
    TimeDomain,
    FrequencyDomain { block_size: usize },
}

impl FastConvolver {
    pub fn new(impulse_response: &[f32], mode: ConvolutionMode) -> Self {
        match mode {
            ConvolutionMode::TimeDomain => {
                let state_size = impulse_response.len() - 1;
                let input_buffer_size = 0;
                FastConvolver {
                    impulse_response: impulse_response.to_vec(),
                    mode,
                }
            }
            ConvolutionMode::FrequencyDomain { block_size } => {
                let state_size = block_size - 1;
                let input_buffer_size = 0;
                FastConvolver {
                    impulse_response: impulse_response.to_vec(),
                    mode,
                }
            }   
        }
    }

    pub fn reset(&mut self) {
        todo!("implement")
    }

    pub fn process(&mut self, input: &[f32], output: &mut [f32]) {
        match self.mode {
            ConvolutionMode::TimeDomain => {
                // Time domain convolution implementation
                self.tdConv(input, output);
            }
            ConvolutionMode::FrequencyDomain { .. } => {
                // Frequency domain convolution implementation
                todo!("implement")
                // self.fdConv(input, output);
            }
        }
    }

    pub fn flush(&mut self, output: &mut [f32]) {
        todo!("implement")
    }

    // Time Domain Convolution
    fn tdConv(&mut self, signal: &[f32], output: &mut [f32]) -> Vec<f32> {
        let kernel = &self.impulse_response; // get impluse response
        let signal_len = signal.len();
        let kernel_len = kernel.len();
        let output_len = signal_len + kernel_len - 1;
    
        for n in 0..output_len {
            let mut sum = 0.0;
            for k in 0..kernel_len {
                if n >= k && n - k < signal_len {
                    sum += signal[n - k] * kernel[k];
                }
            }
            output[n] = sum;
        }
        output.to_vec()
    }

    // Freq Domain Convolution
    fn fdConv(&mut self, signal: &[f32], output: &mut [f32], block_size: usize) {
        todo!("implement")
        // let signal_len = signal.len();
        // let kernel_len = self.impulse_response.len();
        // let output_len = signal_len + kernel_len - 1;

        // let mut fft = RealToComplex::new(real_signal.len());
        // let complex_signal = fft.forward(&real_signal);   

        // let mut signal_fft = vec![Complex::zero(); block_size];
        // let mut kernel_fft = vec![Complex::zero(); block_size];
        // let mut output_fft = vec![Complex::zero(); block_size];
                
        // let mut fft_planner = FftPlanner::new();
        // let fft = fft_planner.plan_fft_forward(block_size);
    
        // // Perform FFT on both signals
        // fft.process_with_scratch(signal, &mut signal_fft);
        // fft.process_with_scratch(self.impulse_response, &mut kernel_fft);

        // // Multiply the transformed input signal and impulse response
        // for i in 0..block_size {
        //     output_fft[i] = signal_fft[i] * kernel_fft[i];
        // }
    
        // // Perform inverse FFT
        // let mut ifft_planner = FftPlanner::new();
        // let ifft = ifft_planner.plan_fft_inverse(block_size);      

        // ifft.process(output_fft.as_mut_slice());
    }

}

// TODO: feel free to define other types (here or in other modules) for your own use


#[cfg(test)]
mod tests {
    use rustfft::num_traits::ToPrimitive;

    use super::*;

    #[test]
    fn complex_signal_reconstruct() {
        let mut signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let length = signal.len();
    
        let mut real_planner = RealFftPlanner::<f64>::new();
        let r2c = real_planner.plan_fft_forward(length);
        let c2r = real_planner.plan_fft_inverse(length);

        // make a vector for storing the spectrum
        let mut spectrum = r2c.make_output_vec();          
        assert_eq!(spectrum.len(), length/2+1);   

        r2c.process(&mut signal, &mut spectrum).unwrap();
        
        // idft
        let mut outdata = c2r.make_output_vec();
        c2r.process(&mut spectrum, &mut outdata).unwrap();

        for elem in outdata.iter_mut() {
            *elem /= length as f64;
        }

        println!("{:?}", outdata);
    }

    #[test]
    fn test_td_convolution1() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let impulse_response = vec![0.5, 1.0, 0.5];
        let mut output = vec![0.0; signal.len() + impulse_response.len() - 1];
        let mut convolver = FastConvolver::new(&impulse_response, ConvolutionMode::TimeDomain);
        convolver.tdConv(&signal, &mut output);

        assert_eq!(output, vec![0.5, 2.0, 4.0, 6.0, 8.0, 7.0, 2.5]);
    }   

    #[test]
    fn test_td_convolution2() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let impulse_response = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let mut output = vec![0.0; signal.len() + impulse_response.len() - 1];
        let mut convolver = FastConvolver::new(&impulse_response, ConvolutionMode::TimeDomain);
        convolver.tdConv(&signal, &mut output);

        assert_eq!(
            output,
            vec![0.1, 0.4, 1.0, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 11.4, 10.6, 8.5, 5.0]
        );        
    }     
}

