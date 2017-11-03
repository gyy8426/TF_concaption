import tensorflow as tf
from tensorflow.contrib import metrics
from tensorflow.contrib.learn import MetricSpec

def accumulate_strings(values, name="strings"):
  """Accumulates strings into a vector.

  Args:
    values: A 1-d string tensor that contains values to add to the accumulator.

  Returns:
    A tuple (value_tensor, update_op).
  """
  tf.assert_type(values, tf.string)
  strings = tf.Variable(
      name=name,
      initial_value=[],
      dtype=tf.string,
      trainable=False,
      collections=[],
      validate_shape=True)
  value_tensor = tf.identity(strings)
  update_op = tf.assign(
      ref=strings, value=tf.concat([strings, values], 0), validate_shape=False)
  return value_tensor, update_op
  
a = ["as asd asd","asd sad","sadasd sd","a a a"]
b = [a,a,a]
print("!!!!!!!!!!b :",b)

c = accumulate_strings(b)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

resluts = sess.run(c)

print("results :",resluts)
