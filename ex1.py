import tensorflow as tf

input_data=[[1,5,3,7,8,10,12],[5, 8, 10, 3, 9, 7,1]]
label_data=[[0,0,0,1,0], [1, 0 ,0, 0, 0]]

INPUT_SIZE=7
HIDDEN1_SIZE=10
HIDDEN2_SIZE=8
CLASSES=5
Learning_Rate=0.05

# shape must be matched to data dimension
x=tf.placeholder(tf.float32, shape=[None, INPUT_SIZE]) #shape=[batchSize, dimension]
y_=tf.placeholder(tf.float32, shape=[None, CLASSES])

tensor_map={x:input_data, y_:label_data}

# Building Model
W_h1 =tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32, name='W_h1') # truncated_normal: Outputs random values from a normal distribution
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32, name='b_h1')

W_h2 =tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32, name='W_h2')
b_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32, name='b_h2')

W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32, name='W_o')
b_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32, name='b_o')

param_list=[W_h1, b_h1, W_h2, b_h2, W_o, b_o]
saver = tf.train.Saver(param_list)


with tf.name_scope('hidden_layer_1') as h1scope:
    hidden1=tf.sigmoid(tf.matmul(x, W_h1) +b_h1, name='hidden1')
with tf.name_scope('hidden_layer_2') as h2scope:
    hidden2=tf.sigmoid(tf.matmul(hidden1,W_h2) + b_h2, name='hidden2')
with tf.name_scope('output_layer') as oscope:
    y=tf.sigmoid(tf.matmul(hidden2, W_o) +b_o, name='y')


# Training
with tf.name_scope('calculate_cost'):
    cost= tf.reduce_sum(-y_*tf.log(y)-(1-y_)*tf.log(1-y), reduction_indices=1)
    cost = tf.reduce_mean(cost)
    tf.scalar_summary('cost/', cost)
    # tf.image_summary('cost', cost)
with tf.name_scope('training'):
    train= tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)
with tf.name_scope('evaluation'):
    # argmax ==> obtaining index
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy/',accuracy)
sess = tf.Session()
# init=tf.global_variables_initializer()
# sess.run(init)
saver.restore(sess,'./tensorflow_checkpoint.ckpt')
merge=tf.merge_all_summaries()

for i in range(1000):
    summary, _, loss, acc = sess.run([merge, train, cost, accuracy], tensor_map)
    # _, loss, acc = sess.run([train, cost, accuracy], tensor_map)
    if i%100==0:
        train_writer = tf.train.SummaryWriter('./summary/', sess.graph)
        train_writer.add_summary(summary, i)
        # saver.save(sess, './tensorflow_checkpoint.ckpt')
        print("==============")
        print("Step: ", i)
        print("Loss: ", loss)
        print("Acc: ", acc)

sess.close()
# tensorboard --logdir=./