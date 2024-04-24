use std::{fs::File, io::Write};

mod ring_buffer;
mod fast_convolver;

use fast_convolver::{ConvolutionMode, FastConvolver};
use rustfft::num_traits::sign;
use std::time::{Instant, Duration};

fn show_info() {
    eprintln!("MUSI-6106 Assignment Executable");
    eprintln!("(c) 2024 Stephen Garrett & Ian Clester");
}

fn block_convolution(
    signal: &[f32],
    impulse_response: &[f32],
    hop_size: usize,
    conv_mode: ConvolutionMode,
) -> Vec<f32> {
    let signal_len = signal.len();
    let ir_len = impulse_response.len();
    let block_size = hop_size + ir_len - 1;

    let mut output_buffer = vec![0.0; signal_len + ir_len - 1];

    let mut convolver = FastConvolver::new(impulse_response, conv_mode);

    let mut start = 0;
    while start + hop_size <= signal_len {
        let mut segment_output = vec![0.0; block_size];
        let segment_x = &signal[start..start + hop_size];
        convolver.process(segment_x, &mut segment_output);

        for i in 0..block_size {
            output_buffer[start + i] += segment_output[i];
        }
        start += hop_size;
    }

    output_buffer
}

fn partitioned_convolution(
    signal: &[f32],
    impulse_response: &[f32],
    block_size: usize,
    conv_mode: ConvolutionMode,
) -> Vec<f32> {
    let signal_len = signal.len();
    let ir_len = impulse_response.len();

    let mut output_buffer = vec![0.0; signal_len + ir_len - 1];

    let num_signal_blocks = (signal_len + block_size - 1) / block_size;
    let num_ir_blocks = (ir_len + block_size - 1) / block_size;

    for i in 0..num_signal_blocks {
        let start = i * block_size;
        let end = std::cmp::min(start + block_size, signal_len);
        let signal_block = &signal[start..end];

        for j in 0..num_ir_blocks {
            let ir_start = j * block_size;
            let ir_end = std::cmp::min(ir_start + block_size, ir_len);
            let ir_block = &impulse_response[ir_start..ir_end];

            let mut convolver = FastConvolver::new(ir_block, conv_mode.clone());
            let mut segment_output = vec![0.0; signal_block.len() + ir_block.len() - 1];
            convolver.process(signal_block, &mut segment_output);

            for k in 0..segment_output.len() {
                output_buffer[start + ir_start + k] += segment_output[k];
            }
        }
    }

    output_buffer
}

fn main() {
    show_info();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!(
            "Usage: {} <Input Signal filename> <Input IR filename> <Output Filename>",
            args[0]
        );
        return;
    }

    // Open the input wave files
    let mut sig_reader = hound::WavReader::open(&args[1]).unwrap();
    let mut ir_reader = hound::WavReader::open(&args[2]).unwrap();
    let spec = sig_reader.spec();
    // create wav writer
    let mut writer = hound::WavWriter::create(&args[3], spec).unwrap();

    let hop_size = 8192;
    let signal_len = sig_reader.duration() as usize;
    let ir_len = ir_reader.duration() as usize;

    let mut impulse_response: Vec<f32> = match ir_reader.spec().sample_format {
        hound::SampleFormat::Int => ir_reader.samples::<i32>().map(|s| s.unwrap() as f32 / i32::MAX as f32).collect(),
        hound::SampleFormat::Float => ir_reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };

    let mut signal: Vec<f32> = match sig_reader.spec().sample_format {
        hound::SampleFormat::Int => sig_reader.samples::<i32>().map(|s| s.unwrap() as f32 / i32::MAX as f32).collect(),
        hound::SampleFormat::Float => sig_reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    };
    // Pad the signal to make it a multiple of hop_size
    let pad_len = hop_size - (signal_len % hop_size);
    signal.extend(std::iter::repeat(0.0).take(pad_len));

    // Time-domain convolution
    let td_start = Instant::now();
    let mut td_output_buffer = block_convolution(&signal, &impulse_response, hop_size, ConvolutionMode::TimeDomain);
    let td_elapsed = td_start.elapsed();

    let mut td_convolver = FastConvolver::new(&impulse_response, ConvolutionMode::TimeDomain);
    let mut td_tail_output = vec![0.0; impulse_response.len() - 1];
    td_convolver.flush(&mut td_tail_output);
    td_output_buffer[signal_len..].copy_from_slice(&td_tail_output);

    // Frequency-domain convolution using partitioned convolution
    let fd_start = Instant::now();
    let block_size = 8192;
    let mut fd_output_buffer = partitioned_convolution(&signal, &impulse_response, block_size, ConvolutionMode::FrequencyDomain { block_size });
    let fd_elapsed = fd_start.elapsed();

    let mut fd_convolver = FastConvolver::new(&impulse_response, ConvolutionMode::FrequencyDomain { block_size });
    let mut fd_tail_output = vec![0.0; impulse_response.len() - 1];
    fd_convolver.flush(&mut fd_tail_output);
    fd_output_buffer[signal_len..].copy_from_slice(&fd_tail_output);

    // Write the scaled samples to the WAV file
    for &sample in &fd_output_buffer {
        let sample_i16 = (sample * (i16::MAX as f32)) as i16;
        writer.write_sample(sample_i16).unwrap();
    }
    // Close the WAV writer to flush the remaining samples
    writer.finalize().unwrap();

    // Print the runtime performance comparison
    println!("Time-domain convolution: {:?}", td_elapsed);
    println!("Frequency-domain convolution (partitioned): {:?}", fd_elapsed);
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
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ir_len = 51;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Generate an input signal with an impulse at sample index 3 (10 samples long)
        let input_len = 10;
        let mut input_signal = vec![0.0; input_len];
        input_signal[3] = 1.0;

        // Create FastConvolver instances for both time-domain and frequency-domain convolution
        let mut td_convolver = FastConvolver::new(&impulse_response, ConvolutionMode::TimeDomain);
        let block_size = ir_len + input_len - 1;
        let mut fd_convolver = FastConvolver::new(
            &impulse_response,
            ConvolutionMode::FrequencyDomain { block_size },
        );

        // Process the input signal and get the output
        let mut td_output = vec![0.0; ir_len + input_len - 1];
        let mut fd_output = vec![0.0; ir_len + input_len - 1];
        td_convolver.process(&input_signal, &mut td_output);
        fd_convolver.process(&input_signal, &mut fd_output);

        // Check the output signal values
        for i in 0..ir_len {
            assert_relative_eq!(td_output[i + 3], impulse_response[i], epsilon = 1e-6);
            assert_relative_eq!(fd_output[i + 3], impulse_response[i], epsilon = 1e-6);
        }
    }
    #[test]
    fn test_flush() {
        // Generate a random impulse response of length 51
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ir_len = 51;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Generate an input signal with an impulse at sample index 3 (10 samples long)
        let input_len = 10;
        let mut input_signal = vec![0.0; input_len];
        input_signal[3] = 1.0;

        // Create FastConvolver instances for both time-domain and frequency-domain convolution
        let mut td_convolver = FastConvolver::new(&impulse_response, ConvolutionMode::TimeDomain);
        let block_size = ir_len + input_len - 1;
        let mut fd_convolver = FastConvolver::new(
            &impulse_response,
            ConvolutionMode::FrequencyDomain { block_size },
        );

        // Process the input signal and get the output
        let mut td_output = vec![0.0; ir_len + input_len - 1];
        let mut fd_output = vec![0.0; ir_len + input_len - 1];
        td_convolver.process(&input_signal, &mut td_output);
        fd_convolver.process(&input_signal, &mut fd_output);

        // Get the tail of the impulse response
        let tail_length = ir_len - 1;
        let mut td_tail_output = vec![0.0; tail_length];
        let mut fd_tail_output = vec![0.0; tail_length];
        td_convolver.flush(&mut td_tail_output);
        fd_convolver.flush(&mut fd_tail_output);

        // Check the tail of the impulse response
        for i in 0..tail_length {
            assert_relative_eq!(td_tail_output[i], td_output[input_len + i], epsilon = 1e-6);
            assert_relative_eq!(fd_tail_output[i], fd_output[input_len + i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_blocksize() {
        // Generate a random impulse response of length 51
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ir_len = 51;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Generate an input signal of length 10000 with an impulse at sample index 3
        let input_len = 10000;
        let mut input_signal = vec![0.0; input_len];
        input_signal[3] = 1.0;

        // Define the block sizes to test
        let block_sizes = [1, 13, 1023, 2048, 1, 17, 5000, 1897];

        for &block_size in &block_sizes {
            // Create FastConvolver instances for both time-domain and frequency-domain convolution
            let mut td_convolver = FastConvolver::new(&impulse_response, ConvolutionMode::TimeDomain);
            let mut fd_convolver = FastConvolver::new(
                &impulse_response,
                ConvolutionMode::FrequencyDomain { block_size },
            );

            // Process the input signal and get the output
            let mut td_output = vec![0.0; ir_len + input_len - 1];
            let mut fd_output = vec![0.0; ir_len + input_len - 1];
            for i in (0..input_len).step_by(block_size) {
                let end = std::cmp::min(i + block_size, input_len);
                td_convolver.process(&input_signal[i..end], &mut td_output[i..i + ir_len + block_size - 1]);
                fd_convolver.process(&input_signal[i..end], &mut fd_output[i..i + ir_len + block_size - 1]);
            }

            // Check the output signal values
            for i in 0..ir_len {
                assert_relative_eq!(td_output[i + 3], impulse_response[i], epsilon = 1e-6);
                assert_relative_eq!(fd_output[i + 3], impulse_response[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_partitioned_identity() {
        // Generate a random impulse response of length 51
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ir_len = 51;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Generate an input signal with an impulse at sample index 3 (10 samples long)
        let input_len = 10;
        let mut input_signal = vec![0.0; input_len];
        input_signal[3] = 1.0;

        // Define the block sizes to test
        let block_sizes = [64, 128, 256, 512, 1024];

        for &block_size in &block_sizes {
            // Perform partitioned convolution
            let conv_mode = ConvolutionMode::FrequencyDomain { block_size };
            let output = partitioned_convolution(&input_signal, &impulse_response, block_size, conv_mode);

            // Check the output signal values
            for i in 0..ir_len {
                assert_relative_eq!(output[i + 3], impulse_response[i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_partitioned_flush() {
        // Generate a random impulse response of length 51
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ir_len = 51;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Generate an input signal with an impulse at sample index 3 (10 samples long)
        let input_len = 10;
        let mut input_signal = vec![0.0; input_len];
        input_signal[3] = 1.0;

        // Define the block sizes to test
        let block_sizes = [64, 128, 256, 512, 1024];

        for &block_size in &block_sizes {
            // Perform partitioned convolution
            let conv_mode = ConvolutionMode::FrequencyDomain { block_size };
            let mut output = partitioned_convolution(&input_signal, &impulse_response, block_size, conv_mode);

            // Create a FastConvolver instance for frequency-domain convolution
            let mut fd_convolver = FastConvolver::new(&impulse_response, conv_mode);

            // Get the tail of the impulse response
            let tail_length = ir_len - 1;
            let mut fd_tail_output = vec![0.0; tail_length];
            fd_convolver.flush(&mut fd_tail_output);

            // Check the tail of the impulse response
            for i in 0..tail_length {
                assert_relative_eq!(fd_tail_output[i], output[input_len + i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_partitioned_blocksize() {
        // Generate a random impulse response of length 51
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let ir_len = 51;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Generate an input signal of length 10000 with an impulse at sample index 3
        let input_len = 10000;
        let mut input_signal = vec![0.0; input_len];
        input_signal[3] = 1.0;

        // Define the block sizes to test
        let block_sizes = [64, 128, 256, 512, 1024];

        for &block_size in &block_sizes {
            // Perform partitioned convolution
            let conv_mode = ConvolutionMode::FrequencyDomain { block_size };
            let output = partitioned_convolution(&input_signal, &impulse_response, block_size, conv_mode);

            // Check the output signal values
            for i in 0..ir_len {
                assert_relative_eq!(output[i + 3], impulse_response[i], epsilon = 1e-6);
            }
        }
    }
}