import torch
import torch.nn.functional as F
from torch.autograd import Function
from itertools import permutations
from torch.distributions import Gumbel
import pdb


def gumbel_like(*args, **kwargs):
    return _gumbel(torch.rand_like(*args, **kwargs))


def gumbel(*args, **kwargs):
    return _gumbel(torch.rand(*args, **kwargs))


def _gumbel(u):
    return -torch.log(-torch.log(u))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return torch.where(
        x >= 10,  # means that y < 1e-4 so O(y^6) <= 1e-24 so we can use series expansion
        -x - y / 2 + y ** 2 / 24 - y ** 4 / 2880,  # + O(y^6), https://www.wolframalpha.com/input/?i=log(1+-+exp(-y))
        torch.log(-torch.expm1(-torch.exp(-x)))  # Hope for the best
    )



def gumbel_with_maximum(phi, T, dim=-1):
    """
    Samples a set of gumbels which are conditioned on having a maximum along a dimension
    phi.max(dim)[0] should be broadcastable with the desired maximum T
    """
    # Gumbel with location phi
    g_phi = phi + gumbel_like(phi)
    Z, argmax = g_phi.max(dim)
    g = _shift_gumbel_maximum(g_phi, T, dim, Z=Z)
    CHECK_VALIDITY = True
    if CHECK_VALIDITY:
        g_inv = _shift_gumbel_maximum(g, Z.unsqueeze(1), dim)
        if not (((g_phi - g_inv) < 1e-2) | (g_phi == g_inv)).all():
            pdb.set_trace()
        ######## increase the threshold originally 1e-3 ########
        assert (((g_phi - g_inv) < 1e-2) | (g_phi == g_inv)).all()
    return g, argmax, g_phi


def _shift_gumbel_maximum(g_phi, T, dim=-1, Z=None):
    if Z is None:
        Z, _ = g_phi.max(dim)
    Z = Z.unsqueeze(1).repeat((1, g_phi.size()[1]))
    T = T.repeat((1, g_phi.size()[1]))
    u = T - g_phi + log1mexp(g_phi - Z)
    return T - F.relu(u) - log1pexp(-u.abs())

def log1mexp(a):
    a = -a.abs()
    zeros_vec = torch.zeros_like(a)
    ones_vec = torch.ones_like(a)
    case1 = torch.where(a >= -0.6931, ones_vec, zeros_vec).byte()
    case2 = torch.where(a < -0.6931, ones_vec, zeros_vec).byte()
    result = torch.zeros_like(a)
    if torch.sum(case1.float()) > 0:
        case1_a = torch.masked_select(a, case1)
        case1_a_ = torch.log(-torch.expm1(case1_a))
        result[case1] = case1_a_
    if torch.sum(case2.float()) > 0:
        case2_a = torch.masked_select(a, case2)
        case2_a_ = safe_log1p(-torch.exp(case2_a))
        result[case2] = case2_a_

    return result
"""

def log1mexp(x):
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    return torch.where(x > -0.693, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))
"""
def log1pexp(a):
    zeros_vec = torch.zeros_like(a)
    ones_vec = torch.ones_like(a)
    case1 = torch.where(a <= -37, ones_vec, zeros_vec).byte()
    case2 = torch.where(a > -37, ones_vec, zeros_vec).byte()
    result = torch.zeros_like(a)
    if torch.sum(case1.float()) > 0:
        case1_a = torch.masked_select(a, case1)
        case1_a_ = torch.exp(case1_a)
        result[case1] = case1_a_
    if torch.sum(case2.float()) > 0:
        case2_a = torch.masked_select(a, case2)
        case2_a_ = safe_log1p(torch.exp(case2_a))
        result[case2] = case2_a_

    return result


class SafeLog1P(Function):

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return torch.log1p(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        # Assume input to be >= 1, if exactly 1, then add small tolerance
        grad_input = grad_output / torch.clamp(1 + input, min=1e-6)  # d/dx log(1+x) = 1/(1+x)

        return grad_input


safe_log1p = SafeLog1P.apply


def all_perms(S, device=None):
    return torch.tensor(list(permutations(S)), device=device)


def log_pl(log_p, dim=-1):
    # Sampling has been done without replacement, compute likelihood without replacement
    # https://math.stackexchange.com/questions/2729561/
    # probability-of-an-unordered-sample-under-weighted-sampling-without-replacement
    # Note that we compute the likelihood for the ordered sample
    a, _ = log_p.max(dim, keepdim=True)
    p = (log_p - a).exp()
    # P = p_1 / 1 * p_2 / (1 - p_1) * p_3 / (1 - p_1 - p_2) ...
    # log P = log p_1 - log(1) + log p_2 - log(1 - p_1) + ...
    #       = sum_i log p_i - sum_i log(1 - sum_j<i p_j)
    # Note that the first term is log_likelihood,
    # and note that sum_j<i p_j = (sum_j<=i p_j) - p_i = cumsum(p_i) - p_i
    # log_partition = partition.log().sum()
    return log_p.sum(dim) - log1mexp(a + (p.cumsum(dim) - p).log()).sum(dim)


def log_pl_rec(log_p, dim=-1):
    """Recursive function of Plackett Luce log probability has better numerical stability
    since 1 - sum_i p_i can get very close to 0, this version never computes sum p_i directly"""
    assert dim == -1
    if log_p.size(-1) == 1:
        return log_p[..., 0]
    return log_p[..., 0] + log_pl_rec(log_p[..., 1:] - log1mexp(log_p[..., 0:1]), dim=dim)


def log_pS_Onfac(log_p):
    return torch.logsumexp(log_pl(all_perms(log_p, device=log_p.device)), -1)


def log_pS_Onfac_rec(log_p):
    return torch.logsumexp(log_pl_rec(all_perms(log_p, device=log_p.device)), -1)


def compute_log_R(log_p, num_points=1000, a=5.):
    # Computes the (log) ratio P(S\{s}|S \subseteq D\{s}) / P(S),
    # where S is an unordered sample under the Plackett-Luce model
    # Additionally computes the (conditional) second order log ratios
    # P(S\{s,s'}|S \subseteq D\{s,s'}) / P(S\{s}|S \subseteq D\{s})
    # Multiplying (or adding in log space) the results gives
    # The unconditional second order log ratios
    # P(S\{s,s'}|S \subseteq D\{s,s'}) / P(S)

    # Constant for numeric stability
    a = log_p.new_tensor(a)

    # Integrals are computed by the trapezoidal rule,
    # which equates to approximating the integral by
    # dx * sum_{i=1}^N (f(i) + f(i-1)) / 2 = dx / 2 * (f(0) + f(N) + 2 * sum_{i = 1}^{N-1} f(i))
    # Since f(0) and f(N) in our integral will be zero, we can just compute
    # dx * sum_{i = 1}^{N-1} f(i)
    # See https://en.wikipedia.org/wiki/Trapezoidal_rule

    # Create range of integration points, (1 ... N-1)/N (bounds are 0 to 1)
    log_v = (torch.arange(1, num_points, out=log_p.new()) / num_points).log()

    # First dim, numerical integration (N - 1)
    # Second dim, batch dimension (B)
    # Third dim, i in S (|S|)
    _q = gumbel_log_survival(-((log_p + a)[None, :, :] + torch.log(-log_v)[:, None, None]))

    # Compute the integrands (N - 1 x B)
    l1me = log1mexp(torch.logsumexp(log_p, -1))
    if torch.isnan(l1me).any() > 0:
        print ("log1mexp(torch.logsumexp(log_p, -1)))")
        adj = torch.logsumexp(log_p, -1) #- 1e-5
        l1me = log1mexp(adj)
        pdb.set_trace()
    q = _q.sum(-1) + (torch.expm1(a + l1me)[None, :] * log_v[:, None])

    # Subtract one factor for element that is left out
    q_without_s = q[..., None] - _q

    # Don't subtract same element twice for diagonals
    skip_diag = 1 - torch.eye(log_p.size(-1), out=log_p.new())[None, None, :, :]
    q_without_ss = q_without_s[..., None] - _q[..., None, :] * skip_diag  # 2nd order

    # To compute the log probabilities, we should add constant a + phi_S, but these cancel out
    sum_S = torch.logsumexp(q, 0)  # e.g. log_P_S = a + phi_S + sum_S
    sum_S_s = torch.logsumexp(q_without_s, 0)
    sum_S_ss = torch.logsumexp(q_without_ss, 0)
    return sum_S_s - sum_S[..., None], sum_S_ss - sum_S_s[..., None]


def all_2nd_order_perms(S, device=None):
    k = S.size(-1)
    ap = all_perms(S, device=device)
    apf = ap[ap[:, 0] < ap[:, 1]].view(k * (k - 1) // 2, -1, k)
    return apf[:, 0, :2], apf[:, :, 2:]


SO_PERM_CACHE = {}



