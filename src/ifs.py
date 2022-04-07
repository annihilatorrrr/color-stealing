import numpy as np

from src import hardcoded
from src.io import load_ifs


def mutate1(ifs):
    mut_prob, noise_std = 0.15, 0.1
    w_b = np.concatenate(ifs[:2], axis=1)

    mut_mask = np.random.binomial(n=1, p=mut_prob, size=w_b.shape)
    if mut_mask.max() == 0:
        idx = np.random.choice(np.size(w_b))
        mut_mask = np.zeros(shape=np.size(w_b))
        mut_mask[idx] = 1
        mut_mask = mut_mask.reshape(w_b.shape)

    noise = np.random.normal(scale=noise_std)
    noise = np.multiply(noise, mut_mask)
    w, b = np.split(w_b + noise, [2], axis=1)

    return w, b


def mutate2(ifs):
    w_b = np.concatenate(ifs[:2], axis=1)
    w_b_shape = w_b.shape

    idx = np.random.choice(np.size(w_b))
    w_b = w_b.flatten()
    w_b[idx] = np.random.uniform(low=-1, high=1)
    w_b = w_b.reshape(w_b_shape)
    w, b = np.split(w_b, [2], axis=1)

    return w, b


def mutate_ifs(ifs):
    all_mutations = [mutate1, mutate2]
    mutate_fn = np.random.choice(all_mutations)
    w, b = mutate_fn(ifs)

    p = np.abs([np.linalg.det(x) for x in w])
    p /= p.sum()

    return w, b, p


def sample_ifs():
    """Creates a random IFS as described in:
    https://arxiv.org/pdf/2110.03091.pdf
    https://arxiv.org/pdf/2101.08515.pdf
    """
    n_ifs = np.random.randint(low=2, high=9)

    theta_phi = np.random.uniform(low=0, high=np.pi, size=[2, n_ifs])
    cos_theta, cos_phi = np.cos(theta_phi)
    sin_theta, sin_phi = np.sin(theta_phi)
    r_theta = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]]).transpose([2, 0, 1])
    r_phi = np.array(
        [[cos_phi, -sin_phi], [sin_phi, cos_phi]]).transpose([2, 0, 1])

    d = np.random.choice([-1., 1.], size=[n_ifs, 2])
    d = np.stack([np.diag(x) for x in d])

    sigma = np.sort(np.random.uniform(size=[n_ifs, 2]))[:, ::-1]
    alpha = sigma[:, 0].sum() + 2 * sigma[:, 1].sum()
    norm = np.random.uniform(low=0.5 * (5 + n_ifs), high=0.5 * (6 + n_ifs))
    sigma = sigma / alpha * norm
    sigma = np.stack([np.diag(x) for x in sigma])

    w = np.einsum(
        'n a b, n b c, n c d, n d e -> n a e', r_theta, sigma, r_phi, d)
    w = w.transpose([0, 2, 1])

    b = np.random.uniform(low=-1, high=1, size=[n_ifs, 1, 2])
    p = np.abs([np.linalg.det(x) for x in w])
    p /= p.sum()

    return w, b, p


def build_ifs(args):

    fixed, load, mutate = args.fixed, args.load, args.mutate

    if fixed and fixed in hardcoded.w:
        w, b, p = hardcoded.w[fixed], hardcoded.b[fixed], hardcoded.p[fixed]
        w, b = w.transpose([0, 2, 1]), b[:, None, :]

    elif load:
        w, b, p = load_ifs(load)

    elif mutate:
        ifs = load_ifs(mutate)
        w, b, p = mutate_ifs(ifs)

    else:
        w, b, p = sample_ifs()

    w, b, p = map(lambda x: x.astype(np.float32), [w, b, p])

    return w, b, p
