import math
import mlx.core as mx
import mlx.nn as nn


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        return nn.init.normal(mean, std)(m)
    else:
        return m


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
    """KL(P||Q)"""
    kl = (logs_q - logs_p) - 0.5
    kl += 0.5 * (mx.exp(2.0 * logs_p) + ((m_p - m_q) ** 2)) * mx.exp(-2.0 * logs_q)
    return kl


def rand_gumbel(shape):
    """Sample from the Gumbel distribution, protect from overflows."""
    uniform_samples = mx.random.uniform(shape=shape) * 0.99998 + 0.00001
    return -mx.log(-mx.log(uniform_samples))


def rand_gumbel_like(x):
    g = rand_gumbel(x.size()).astype(dtype=x.dtype)
    return g


def slice_segments(x, ids_str, segment_size=4):
    ret = mx.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = ids_str[i]
        idx_end = idx_str + segment_size
        ret[i] = x[i, :, idx_str:idx_end]
    return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
    b, d, t = x.size()
    if x_lengths is None:
        x_lengths = t
    ids_str_max = x_lengths - segment_size + 1
    ids_str = (mx.random.uniform(shape=[b]) * ids_str_max).astype(dtype=mx.int64)
    ret = slice_segments(x, ids_str, segment_size)
    return ret, ids_str


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    position = mx.arange(length, dtype=mx.float32)
    num_timescales = channels // 2
    log_timescale_increment = math.log(float(max_timescale) / float(min_timescale)) / (num_timescales - 1)
    inv_timescales = min_timescale * mx.exp(mx.arange(num_timescales, dtype=mx.float32) * -log_timescale_increment)
    scaled_time = position.expand_dims(0) * inv_timescales.expand_dims(1)
    signal = mx.concatenate([mx.sin(scaled_time), mx.cos(scaled_time)], 0)
    signal = mx.pad(signal, [0, 0, 0, channels % 2])
    signal = signal.reshape(1, channels, length)
    return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal.astype(dtype=x.dtype)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
    b, channels, length = x.size()
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return mx.concatenate([x, signal.astype(dtype=x.dtype)], axis)


def subsequent_mask(length):
    mask = mx.tril(mx.ones(length, length)).expand_dims(0).expand_dims(0)
    return mask


def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = mx.tanh(in_act[:, :n_channels_int, :])
    s_act = mx.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def shift_1d(x):
    x = mx.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
    return x


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = mx.arange(max_length, dtype=length.dtype)
    return x.expand_dims(0) < length.expand_dims(1)


def generate_path(duration, mask):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = mx.cumsum(duration, -1)

    cum_duration_flat = cum_duration.reshape(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).astype(mask.dtype)
    path = path.reshape(b, t_x, t_y)
    path = path - mx.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.expand_dims(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, mx.array):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
