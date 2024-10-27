import os
import time
import pickle
import argparse
import tensorflow as tf
import numpy as np
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d
import utils
import models

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i',
                       required=True,
                       dest='input_dir',
                       help='Trained model directory. The --output-dir value used for training.')
    
    parser.add_argument('--checkpoint', '-c',
                       required=True,
                       dest='checkpoint',
                       help='Model checkpoint to use for sampling. Expects a .ckpt file.')
    
    parser.add_argument('--output', '-o',
                       default='samples.txt',
                       help='File path to save generated samples to (default: samples.txt)')
    
    parser.add_argument('--num-samples', '-n',
                       type=int,
                       default=1000000,
                       dest='num_samples',
                       help='The number of password samples to generate (default: 1000000)')
    
    parser.add_argument('--batch-size', '-b',
                       type=int,
                       default=64,
                       dest='batch_size',
                       help='Batch size (default: 64).')
    
    parser.add_argument('--seq-length', '-l',
                       type=int,
                       default=10,
                       dest='seq_length',
                       help='The maximum password length. Use the same value that you did for training. (default: 10)')
    
    parser.add_argument('--layer-dim', '-d',
                       type=int,
                       default=128,
                       dest='layer_dim',
                       help='The hidden layer dimensionality for the generator. Use the same value that you did for training (default: 128)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        parser.error(f'"{args.input_dir}" folder doesn\'t exist')
    
    if not os.path.exists(args.checkpoint + '.index'):
        parser.error(f'Checkpoint file "{args.checkpoint}" doesn\'t exist')
    
    charmap_path = os.path.join(args.input_dir, 'charmap.pickle')
    inv_charmap_path = os.path.join(args.input_dir, 'inv_charmap.pickle')
    
    if not os.path.exists(charmap_path):
        parser.error(f'Character map file "{charmap_path}" doesn\'t exist')
    
    if not os.path.exists(inv_charmap_path):
        parser.error(f'Inverse character map file "{inv_charmap_path}" doesn\'t exist')
    
    return args

def load_charmaps(input_dir):
    with open(os.path.join(input_dir, 'charmap.pickle'), 'rb') as f:
        charmap = pickle.load(f, encoding='latin1')
    
    with open(os.path.join(input_dir, 'inv_charmap.pickle'), 'rb') as f:
        inv_charmap = pickle.load(f, encoding='latin1')
    
    return charmap, inv_charmap

def generate_samples(session, fake_inputs):
    try:
        samples = session.run(fake_inputs)
        print("Generated samples shape:", samples.shape)
        samples = np.argmax(samples, axis=2)
        print("Argmax samples shape:", samples.shape)
        return samples
    except Exception as e:
        print("Error during sample generation:", str(e))
        raise

def decode_samples(samples, inv_charmap):
    try:
        decoded_samples = []
        for i in range(len(samples)):
            try:
                decoded = [inv_charmap[s] for s in samples[i]]
                # Filter out non-ASCII characters and invalid characters
                decoded = [c for c in decoded if ord(c) < 128 and c.isprintable()]
                if decoded:  # Only add non-empty samples
                    decoded_samples.append(tuple(decoded))
            except Exception as e:
                print(f"Warning: Skipping sample {i} due to decoding error: {e}")
                continue
                
        if decoded_samples:
            print("Decoded first sample:", "".join(decoded_samples[0]))
        return decoded_samples
    except Exception as e:
        print("Error during sample decoding:", str(e))
        raise

def save_samples(samples, output_file):
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for s in samples:
                try:
                    # Convert tuple to string and filter out backticks
                    password = "".join(s).replace('`', '')
                    # Only save non-empty passwords with printable ASCII characters
                    if password and all(ord(c) < 128 and c.isprintable() for c in password):
                        f.write(password + "\n")
                except Exception as e:
                    print(f"Warning: Skipping password due to encoding error: {e}")
                    continue
        print(f"Saved {len(samples)} samples to {output_file}")
    except Exception as e:
        print("Error during sample saving:", str(e))
        raise

def main():
    print("Starting password generation...")
    args = parse_args()
    print("Arguments parsed successfully")
    
    charmap, inv_charmap = load_charmaps(args.input_dir)
    print(f"Loaded character maps. Vocab size: {len(charmap)}")
    
    # Configure TensorFlow
    print("Configuring TensorFlow...")
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    
    # Create the session
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.compat.v1.Session(config=config)
    print("TensorFlow session created")
    
    try:
        # Create the generator inputs
        print("Creating generator...")
        fake_inputs = models.Generator(
            args.batch_size,
            args.seq_length,
            args.layer_dim,
            len(charmap)
        )
        print("Generator created successfully")
        
        # Initialize variables
        print("Initializing variables...")
        sess.run(tf.compat.v1.global_variables_initializer())
        print("Variables initialized")
        
        # Create saver and restore
        print(f"Restoring model from checkpoint: {args.checkpoint}")
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, args.checkpoint)
        print("Model restored successfully")
        
        samples_buffer = []
        then = time.time()
        start = time.time()
        
        num_batches = int(np.ceil(args.num_samples / args.batch_size))
        print(f"Will generate {num_batches} batches of size {args.batch_size}")
        
        for i in range(num_batches):
            print(f"\nGenerating batch {i+1}/{num_batches}...")
            # Generate and decode samples
            raw_samples = generate_samples(sess, fake_inputs)
            decoded_samples = decode_samples(raw_samples, inv_charmap)
            samples_buffer.extend(decoded_samples)
            
            if i % 1000 == 0 and i > 0:
                # Save accumulated samples
                save_samples(samples_buffer, args.output)
                samples_buffer = []  # Clear buffer
                
                print(f"Wrote {1000 * args.batch_size} samples to {args.output} in {time.time() - then:.2f} seconds. {i * args.batch_size} total.")
                then = time.time()
        
        # Save any remaining samples
        if samples_buffer:
            save_samples(samples_buffer, args.output)
        
        print(f"Finished in {time.time() - start:.2f} seconds")
    
    except Exception as e:
        print("Error during execution:", str(e))
        raise
    finally:
        print("Closing TensorFlow session...")
        sess.close()

if __name__ == '__main__':
    main()