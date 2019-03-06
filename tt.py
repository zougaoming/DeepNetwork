import tensorflow as tf
import numpy as np


class Batch2ConvMatrix:
    def __init__(self, stride, kernel_h, kernel_w):
        self.stride = stride
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w

        self.x = None
        self.conv_size = None

    def __call__(self, x):
        self.x = x
        x_nums, x_channels, x_height, x_width = np.shape(self.x)

        conv_height = int((x_height - self.kernel_h) / self.stride) + 1
        conv_width = int((x_width - self.kernel_w) / self.stride) + 1

        scan = np.zeros((x_nums, conv_height, conv_width,
                         x_channels, self.kernel_h, self.kernel_w))

        for n in range(x_nums):
            for h in range(conv_height):
                for w in range(conv_width):
                    for c in range(x_channels):
                        start_h = h * self.stride
                        start_w = w * self.stride
                        end_h = start_h + self.kernel_h
                        end_w = start_w + self.kernel_w

                        scan[n, h, w, c] = \
                            x[n, c, start_h:end_h, start_w:end_w]

        conv_matrix = scan.reshape(x_nums * conv_height * conv_width, -1)
        self.conv_size = [x_nums, x_channels, conv_height, conv_width]
        return conv_matrix

    def backward(self, dx2m):
        dx = np.zeros_like(self.x)
        kh = self.kernel_h
        kw = self.kernel_w
        xn, xc, ch, cw = self.conv_size

        dx2m = dx2m.reshape((xn, ch, cw, xc, kh, kw))

        for n in range(xn):
            for c in range(xc):
                for h in range(ch):
                    for w in range(cw):
                        start_h = h * self.stride
                        start_w = w * self.stride
                        end_h = start_h + self.kernel_h
                        end_w = start_w + self.kernel_w

                        dx[n, c][start_h:end_h, start_w:end_w] \
                            += dx2m[n, h, w, c]

        return dx


class Conv2d:
    def __init__(self, stride, weight=None, bias=None):
        self.stride = stride
        self.weight = weight
        self.bias = bias

        self.b2m = None
        self.x2m = None
        self.w2m = None

        self.dw = None
        self.db = None

    def __call__(self, x):
        wn, wc, wh, ww = np.shape(self.weight)

        if self.b2m is None:
            self.b2m = Batch2ConvMatrix(self.stride, wh, ww)

        x2m = self.b2m(x)
        w2m = self.weight.reshape(wn, -1)
        xn, xc, oh, ow = self.b2m.conv_size

        out_matrix = np.matmul(x2m, w2m.T) + self.bias

        out = out_matrix.reshape((xn, oh, ow, wn))

        self.x2m = x2m
        self.w2m = w2m

        out = out.transpose((0, 3, 1, 2))
        return out

    def backward(self, d_loss):
        on, oc, oh, ow = np.shape(d_loss)

        d_loss = d_loss.transpose((0, 2, 3, 1))
        d_loss = d_loss.reshape((on * oh * ow, -1))

        dw = np.matmul(d_loss.T, self.x2m)
        self.dw = dw.reshape(np.shape(self.weight))
        self.db = np.sum(d_loss, axis=0)

        dx2m = np.matmul(d_loss, self.w2m)
        dx = self.b2m.backward(dx2m)
        return dx






tf.enable_eager_execution()
tf.set_random_seed(123)

np.random.seed(123)
np.set_printoptions(6, suppress=True, linewidth=120)

x_numpy = np.random.random((2,5, 7,3))
x_tf = tf.constant(x_numpy)

conv_tf = tf.layers.Conv2D(
    filters=4, kernel_size=3, strides=(2, 2),
    data_format="channels_last")

with tf.GradientTape(persistent=True) as t:
    t.watch(x_tf)
    y_tf = conv_tf(x_tf)

conv_numpy = Conv2d(
    stride=2,
    weight=conv_tf.get_weights()[0].transpose(3, 2, 0, 1),
    bias=conv_tf.get_weights()[1])

x_numpy = x_numpy.transpose(0,3,1,2)
y_numpy = conv_numpy(x_numpy)

dy_numpy = np.random.random(y_numpy.shape)
dy_tf = tf.constant(dy_numpy)

dx_numpy = conv_numpy.backward(dy_numpy)

dy_dx = t.gradient(y_tf, x_tf, dy_tf)
dw_tf, db_tf = t.gradient(y_tf, conv_tf.weights, dy_tf)

print("y_numpy\n", y_numpy[0][0])
print("y_tf\n", y_tf.numpy()[0][0])

print("dx_numpy\n", dx_numpy[0][0])
print("dx_tf\n", dy_dx.numpy()[0][0])

print("dw_numpy\n", conv_numpy.dw[0][0])
print("dw_tf\n", dw_tf.numpy()[0][0])

print("db_numpy\n", conv_numpy.db)
print("db_tf\n", db_tf.numpy())

