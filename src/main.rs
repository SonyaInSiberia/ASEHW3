use std::{fs::File, io::Write};

mod ring_buffer;
mod fast_convolver;

use fast_convolver::{ConvolutionMode, FastConvolver};

fn show_info() {
    eprintln!("MUSI-6106 Assignment Executable");
    eprintln!("(c) 2024 Stephen Garrett & Ian Clester");
}

fn main() {
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
    let hop_size = ir_reader.duration() / 2;
    let block_size = (hop_size + ir_reader.duration() - 1) as usize;   

    // create final output buffer
    let mut output_buffer = vec![
        0.0 as f32; 
        (sig_reader.duration() + ir_reader.duration() + hop_size) as usize
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

        fd_convolver.process(&segment_x, &mut segment_output);

        for i in 0..hop_size {
            output_buffer[(start + i) as usize] += segment_output[i as usize];
        }

        start += hop_size;
    }

    // Scale the samples from f32 to i16 range
    let scaled_output: Vec<i16> = output_buffer.iter().map(|&x| (x * i16::MAX as f32) as i16).collect();
    // Write the scaled samples to the WAV file
    for sample in scaled_output {
        writer.write_sample(sample).unwrap();
    }
    // Close the WAV writer to flush the remaining samples
    writer.finalize().unwrap();


}
