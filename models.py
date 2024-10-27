import tensorflow as tf
import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d

def Generator(batch_size, seq_length, layer_dim, output_dim):
    # [batch_size, noise_dim]
    noise = tf.random.normal([batch_size, 128])
    
    # [batch_size, seq_length * layer_dim]
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_length*layer_dim, noise)
    
    # Reshape to [batch_size, seq_length, layer_dim]
    output = tf.reshape(output, [-1, seq_length, layer_dim])
    
    # Convert to channel-first format [batch_size, layer_dim, seq_length]
    output = tf.transpose(output, [0, 2, 1])
    
    # Apply 1D convolutions
    output = lib.ops.conv1d.Conv1D('Generator.1', layer_dim, layer_dim, 5, output)
    output = tf.nn.relu(output)
    
    output = lib.ops.conv1d.Conv1D('Generator.2', layer_dim, layer_dim, 5, output)
    output = tf.nn.relu(output)
    
    output = lib.ops.conv1d.Conv1D('Generator.3', layer_dim, layer_dim, 5, output)
    output = tf.nn.relu(output)
    
    output = lib.ops.conv1d.Conv1D('Generator.4', layer_dim, layer_dim, 5, output)
    output = tf.nn.relu(output)
    
    output = lib.ops.conv1d.Conv1D('Generator.5', layer_dim, output_dim, 1, output)
    
    # Convert back to batch_size, seq_length, output_dim
    output = tf.transpose(output, [0, 2, 1])
    
    # Apply softmax
    output = tf.nn.softmax(output, axis=-1)
    
    return output

def Discriminator(inputs, seq_length, layer_dim, input_dim):
    # Convert to channel first [batch_size, input_dim, seq_length]
    output = tf.transpose(inputs, [0, 2, 1])
    
    output = lib.ops.conv1d.Conv1D('Discriminator.1', input_dim, layer_dim, 5, output)
    output = tf.nn.leaky_relu(output)
    
    output = lib.ops.conv1d.Conv1D('Discriminator.2', layer_dim, layer_dim, 5, output)
    output = tf.nn.leaky_relu(output)
    
    output = lib.ops.conv1d.Conv1D('Discriminator.3', layer_dim, layer_dim, 5, output)
    output = tf.nn.leaky_relu(output)
    
    output = lib.ops.conv1d.Conv1D('Discriminator.4', layer_dim, layer_dim, 5, output)
    output = tf.nn.leaky_relu(output)
    
    output = lib.ops.conv1d.Conv1D('Discriminator.5', layer_dim, 1, 1, output)
    
    # Reshape to [batch_size]
    output = tf.reshape(output, [-1])
    
    return output