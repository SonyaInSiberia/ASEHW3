use std::{fs::File, io::Write};

mod ring_buffer;
mod fast_convolver;

use fast_convolver::{ConvolutionMode, FastConvolver};
use std::time::{Instant, Duration};

fn show_info() {
    eprintln!("MUSI-6106 Assignment Executable");
    eprintln!("(c) 2024 Stephen Garrett & Ian Clester");
}

fn main() {
    // Time the performance
    let now = Instant::now();  

    show_info();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <Input Signal filename> <Input IR filename> <Output Filename>", args[0]);
        return
    }

    // Open the input wave file
    let mut sig_reader = hound::WavReader::open(&args[1]).unwrap();
    let mut ir_reader = hound::WavReader::open(&args[2]).unwrap();
    let spec = sig_reader.spec();
    // create wav writer
    let mut writer = hound::WavWriter::create(&args[3], spec).unwrap();    

    //---------------- block convolution ----------------//
    let hop_size = 8192;
    let block_size = (hop_size + ir_reader.duration()) as usize;   

    // create final output buffer
    let mut output_buffer = vec![
        0.0 as f32; 
        (sig_reader.duration() + ir_reader.duration() - 1) as usize
    ];

    // save full Sigal and IR to vec
    let mut impulse_response: Vec<f32> = Vec::new();
    for sample in ir_reader.samples::<f32>() {
        let sample = sample.unwrap();
        impulse_response.push(sample);
    }
    let mut signal = Vec::new();
    for sample in sig_reader.samples::<f32>() {
        let sample = sample.unwrap();
        signal.push(sample);
    }

    // fd convolution setup
    let mut fd_convolver = FastConvolver::new(
        &impulse_response, 
        ConvolutionMode::FrequencyDomain { block_size }
    );
    // td convolution setup
    let mut td_convolver = FastConvolver::new(
        &impulse_response, 
        ConvolutionMode::TimeDomain
    );    

    let mut start = 0;
    while start + hop_size < sig_reader.duration() {
        println!("{}", start);
        // fft_len = hop_size + ir.len() - 1
        let mut segment_output = vec![0.0; block_size];

        // block the signal with size of hopsize
        let mut segment_x = vec![0.0; hop_size as usize];
        for i in start..start+hop_size {
            segment_x.push(signal[i as usize]);
        }

        /////// Switch td or fd here ///////
        // fd_convolver.process(&segment_x, &mut segment_output);
        td_convolver.process(&segment_x, &mut segment_output);

        for i in 0..hop_size {
            output_buffer[(start + i) as usize] += segment_output[i as usize];
        }

        start += hop_size;
    }

    // Write the scaled samples to the WAV file
    for sample in output_buffer {
        let sample_i16 = (sample * (i16::MAX as f32)) as i16;
        writer.write_sample(sample_i16).unwrap();
    }
    // Close the WAV writer to flush the remaining samples
    writer.finalize().unwrap();


    // Measure elapsed time
    let elapsed = now.elapsed();
    let elapsed_secs = elapsed.as_secs();
    println!("Program ran for {} seconds.", elapsed_secs);    

}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use std::time::{Instant, Duration};

    #[test]
    fn test_block() {
        // Time the performance
        let now = Instant::now();

        // Generate a random impulse response of length 51
        let mut rng = ChaCha8Rng::seed_from_u64(9418); // set random seed
        let ir_len = 1000;
        let sig_len = 10000;
        let impulse_response: Vec<f32> = (0..ir_len).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let signal: Vec<f32> = (0..sig_len).map(|_| rng.gen_range(-1.0..1.0)).collect();

        //---------------- block convolution ----------------//
        let hop_size = 4;
        let block_size = (hop_size + ir_len) as usize;   

        // create final output buffer
        let mut output_buffer = vec![
            0.0 as f32; 
            (sig_len + ir_len - 1) as usize
        ];

        // fd convolution setup
        let mut fd_convolver = FastConvolver::new(
            &impulse_response, 
            ConvolutionMode::FrequencyDomain { block_size }
        );
        // td convolution setup
        let mut td_convolver = FastConvolver::new(
            &impulse_response, 
            ConvolutionMode::TimeDomain
        );

        let mut start = 0;
        while start + hop_size < sig_len {
            // fft_len = hop_size + ir.len() - 1
            let mut segment_output = vec![0.0; block_size];

            // block the signal with size of hopsize
            let mut segment_x = vec![0.0; hop_size as usize];
            for i in start..start+hop_size {
                segment_x.push(signal[i as usize]);
            }

            // fd_convolver.process(&segment_x, &mut segment_output);
            td_convolver.process(&segment_x, &mut segment_output);

            for i in 0..hop_size {
                output_buffer[(start + i) as usize] += segment_output[i as usize];
            }

            start += hop_size;
        }

        // Measure elapsed time
        let elapsed = now.elapsed();

        // Convert elapsed time to seconds
        let elapsed_secs = elapsed.as_secs();
        println!("Program ran for {} seconds.", elapsed_secs);     
    }
}