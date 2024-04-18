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
            ConvolutionMode::FrequencyDomain { block_size } => {
                // Frequency domain convolution implementation
                self.fdConv(input, output, block_size);
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
    fn fdConv(&mut self, signal: &[f32], output: &mut [f32], block_size: usize) -> Vec<f32> {
        let mut real_planner = RealFftPlanner::<f32>::new();
        let r2c = real_planner.plan_fft_forward(block_size);
        let c2r = real_planner.plan_fft_inverse(block_size);

        // make a vector for storing the spectrum
        let mut signal_fft: Vec<Complex<f32>> = r2c.make_output_vec();
        let mut kernel_fft: Vec<Complex<f32>> = r2c.make_output_vec();

        // zero-padding signal
        let mut padded_signal = if signal.len() < block_size {
            let num_zeros = block_size - signal.len();
            let mut temp = signal.to_vec();
            temp.extend(vec![0.0; num_zeros]);
            temp
        } else {
            signal.to_vec()
        };

        // zero-padding IR
        let mut padded_ir = if self.impulse_response.len() < block_size {
            let num_zeros = block_size - self.impulse_response.len();
            let mut temp = self.impulse_response.to_vec();
            temp.extend(vec![0.0; num_zeros]);
            temp
        } else {
            self.impulse_response.to_vec()
        };

        // fft
        r2c.process(&mut padded_signal, &mut signal_fft).unwrap();
        r2c.process(&mut padded_ir, &mut kernel_fft).unwrap();

        // Frequency domain convolution
        for i in 0..signal_fft.len() {
            signal_fft[i] *= kernel_fft[i];
        }

        // IFFT
        c2r.process(&mut signal_fft, output).unwrap();
        for i in 0..output.len() {
            output[i] = output[i] / output.len() as f32;
        }

        output.to_vec()

    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_identity() {
        // Generate a random impulse response of length 51
        let mut rng = ChaCha8Rng::seed_from_u64(9418); // set random seed
        let ir_len = 51;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Generate an input signal with an impulse at sample index 3 (10 samples long)
        let input_len = 10;
        let mut input_signal = vec![0.0; input_len];
        input_signal[3] = 1.0;

        // Create a FastConvolver instance with the random impulse response
        let mut td_convolver = FastConvolver::new(&impulse_response, ConvolutionMode::TimeDomain);
        let block_size = ir_len + input_len - 1;
        let mut fd_convolver = FastConvolver::new(&impulse_response, ConvolutionMode::FrequencyDomain { block_size });

        // Process the input signal and get the output
        let mut td_output = vec![0.0; block_size];
        let mut fd_output = vec![0.0; block_size];
        td_convolver.process(&input_signal, &mut td_output);
        fd_convolver.process(&input_signal, &mut fd_output);

        // Check the first 10 samples of the output
        let expected_output = vec![
            0.0,
            0.0, 
            0.0, 
            -0.5180299, 
            0.7389345, 
            -0.57418156, 
            0.9897876, 
            -0.6168113, 
            -0.48569036, 
            0.23319936, 
        ];

        for (expected, actual) in expected_output.iter().zip(td_output.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
        for (expected, actual) in expected_output.iter().zip(fd_output.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }        
    }    



    ////////////////////////////////////
    ////   Some extra tests below   ////
    ////////////////////////////////////
    #[test]
    fn test_fd_convolution1() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let impulse_response = vec![0.5, 1.0, 0.5];
        let fft_length = signal.len() + impulse_response.len() - 1;
        let block_size = fft_length;
        let mut output = vec![0.0; fft_length];
    
        let mut convolver = FastConvolver::new(&impulse_response, ConvolutionMode::FrequencyDomain { block_size });
        convolver.fdConv(&signal, &mut output, block_size);
    
        let expected_output = vec![0.5, 2.0, 4.0, 6.0, 8.0, 7.0, 2.5];
        for (expected, actual) in expected_output.iter().zip(output.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    } 

    #[test]
    fn test_fd_convolution2() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let impulse_response = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let fft_length = signal.len() + impulse_response.len() - 1;
        let block_size = fft_length;
        let mut output = vec![0.0; fft_length];
    
        let mut convolver = FastConvolver::new(&impulse_response, ConvolutionMode::FrequencyDomain { block_size });
        convolver.fdConv(&signal, &mut output, block_size);

        let expected_output = vec![0.1, 0.4, 1.0, 2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 11.4, 10.6, 8.5, 5.0];

        for (expected, actual) in expected_output.iter().zip(output.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }     
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
}

