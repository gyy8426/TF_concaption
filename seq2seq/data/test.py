import matplotlib.pyplot as plt;  
import tensorflow as tf;  
  
image_raw_data_jpg = tf.gfile.FastGFile('haha.jpg', 'r').read()  
with tf.Session() as sess:  
    img_data_jpg = tf.image.decode_jpeg(image_raw_data_jpg) 
    img_data_jpg = tf.image.convert_image_dtype(img_data_jpg, dtype=tf.uint8)  
    encode_image_jpg = tf.image.encode_jpeg(img_data_jpg) 
    encode_image_png = tf.image.encode_png(img_data_jpg)
    print("tensorflow image type: ", type(encode_image_jpg.eval()))
    