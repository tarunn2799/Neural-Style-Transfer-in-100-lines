import tensorflow as tf, scipy.misc, numpy as np
content = scipy.misc.imread('content.jpg').astype(np.float)
style = scipy.misc.imread('style.jpg').astype(np.float)
content_layers = 'conv4_2'
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
cw, sw, tvw, lr, iter = 30, 15, 0, 1e0, 1000
style_weight_layer = sw/len(style_layers)
 
def style_layer_loss(style_out, target_out, layer):
        N = get_shape(target_out[layer])[3] # number of feature maps
        M = get_shape(target_out[layer])[1] * get_shape(target_out[layer])[2] # dimension of each feature map
        style_gram = tf.matmul(tf.reshape(style_out[layer], [-1, get_shape(style_out[layer])[3]]), tf.reshape(style_out[layer], [-1, get_shape(style_out[layer])[3]]), transpose_a=True)
        target_gram = tf.matmul(tf.reshape(target_out[layer], [-1, get_shape(target_out[layer])[3]]), tf.reshape(target_out[layer], [-1, get_shape(target_out[layer])[3]]), transpose_a=True)
        st_loss = tf.multiply(tf.reduce_sum(tf.square(tf.subtract(target_gram, style_gram))), 1./((N**2) * (M**2)))
        st_loss = tf.multiply(st_loss, style_weight_layer, name='style_loss')
        return st_loss
def get_shape(inp):
    if type(inp) == type(np.array([])):
        return inp.shape
    else:
        return [i.value for i in inp.get_shape()]
def get_model(input_image):
    def conv2d(x, W, stride, padding="SAME"):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    def max_pool(x, k_size, stride, padding="VALID"):
        return tf.nn.avg_pool(x, ksize=[1, k_size, k_size, 1],
                strides=[1, stride, stride, 1], padding=padding)
    def get_type(layer):
        if 'conv' in layer:
            return 'conv'
        if 'pool' in layer:
            return 'pool'
        if 'relu' in layer:
            return 'relu'
    model = {}
    net_data = np.load("vgg16_weights.npz")
    layers = []
    layers.extend(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'maxpool1'])
    layers.extend(['conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'maxpool2'])
    layers.extend(['conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'maxpool3'])
    layers.extend(['conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'maxpool4'])
    layers.extend(['conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'maxpool5'])
    current = input_image
    for layer in layers:
        layer_type = get_type(layer)
        if layer_type == 'conv':
            W_conv = tf.constant(net_data[layer+'_W'])
            b_conv = tf.constant(net_data[layer+'_b'])
            conv_out = tf.nn.conv2d(current, W_conv, strides=[1, 1, 1, 1], padding='SAME')
            current = tf.nn.bias_add(conv_out, b_conv)
        elif layer_type == 'pool':
            current = tf.nn.avg_pool(current, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
        model[layer] = current
    return model
def process(image, proc = 'pre'):
    image = image[...,::-1]
    if(proc == 'pre'):
        return (image - np.array([104,117, 123]))
    else:
        return (image + np.array([123, 117, 104]))
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
content_pre = np.array([process(content, proc = 'pre')])
    img = tf.placeholder('float', shape=content_pre.shape)
    model = get_model(img)
    c_out = sess.run(model[content_layers], feed_dict={img:content_pre})
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    style_pre = np.array([process(style, proc = 'pre')])
    img = tf.placeholder('float', shape=style_pre.shape)
    model = get_model(img)
    s_out = sess.run({s_l:model[s_l] for s_l in style_layers}, feed_dict = {img:style_pre})
g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    target_pre_var = tf.Variable(tf.random_normal((1,)+content.shape))
    model = get_model(target_pre_var)
    contl = tf.multiply(tf.reduce_sum(tf.square(tf.subtract(model[content_layers], c_out))), cw)
    stylel = []
    for s_l in style_layers:
        loss = style_layer_loss(s_out, model, s_l)
        stylel.append(loss)
    batch, width, height, channels = get_shape(target_pre_var)
    width_var = tf.nn.l2_loss(tf.subtract(target_pre_var[:,:width-1,:,:], target_pre_var[:,1:,:,:]))
    height_var = tf.nn.l2_loss(tf.subtract(target_pre_var[:,:,:height-1,:], target_pre_var[:,:,1:,:]))
    tvl = tvw*tf.add(width_var, height_var)
    totL = contl + tf.add_n(stylel) + tvl
    t_s = tf.train.AdamOptimizer(lr).minimize(totL)
    sess.run(tf.global_variables_initializer())
    min_loss, best = float("inf"), None
    for i in range(iter):
        t_s.run()
        if(i%100 == 0):
            if(totL.eval() < min_loss):
                min_loss = totL.eval()
                best = target_pre_var.eval()
    save = np.clip(process(best.squeeze(), proc = 'post'), 0, 255).astype(np.uint8)
    scipy.misc.imsave('output.jpg', save)