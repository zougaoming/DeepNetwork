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
