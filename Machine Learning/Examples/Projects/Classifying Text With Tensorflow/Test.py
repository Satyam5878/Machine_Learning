import tensorflow as tf


my_graph = tf.Graph()
with tf.Session(graph=my_graph) as sess:
    """
    x = tf.constant([1,3,6])
    y = tf.constant([1,1,1])
    
    op = tf.add(x,y)
    
    result = sess.run(fetches=op)
    print(result)

    """
    x = tf.constant([[1.,1.],[2.,2.]])
    y = tf.reduce_mean(x)
    sess.run(fetches=y)
    print(y)
