###
### Start again from here
###

# Standard imports
import numpy as np

# Specialized imports
import scipy.sparse as sp
import itertools
import time
import pdb
import matplotlib.pyplot as plt

# This is needed to fix tensor product order error throughout
def flip_parts_in_spec_str(spec_str):
    parts = spec_str.split('+')
    flipped_parts = [part[::-1] for part in parts]
    return '+'.join(flipped_parts)

def make_all_seqs(L, alphabet='ACGT', max_num_seqs=1000):
    """
    Creates all sequences of a given length from a given alphabet up to some
    maximum number.
    :param L: (int > 0)
        Sequence length.
    :param alphabet: (iterable over chars)
        Alphabet from which to generate sequences.
    :param max_num_seqs: (int > 0)
        Maximum number of sequences to generate.
    :return:
        List of generated sequences.
    """
    alphabet_list = list(alphabet)
    iterable = itertools.product(alphabet_list, repeat=L)
    return [''.join(seq) for seq in itertools.islice(iterable, max_num_seqs)]


def make_random_seqs(L, num_seqs, alphabet):
    """
    Creates a given number of random sequences.
    :param L: (int > 0)
        Sequence length.
    :param num_seqs: (int > 0)
        Number of sequences to generate.
    :param alphabet: (iterable over chars)
        Alphabet from which to generate sequences.
    :return:
        List of generated sequences.
    """
    return [''.join(row) for row in
            np.random.choice(a=list(alphabet), size=[num_seqs, L])]


def seq_to_x_ohe(seq, ohe_spec, alphabet):
    """
    Creates a one-hot encoding of a sequence. Much faster than seq_to_x_sim.
    :param seq: (str)
        Sequence to encode.
    :param ohe_spec:
        Specification string for one-hot encoding.
    :param alphabet: (iterable over chars)
        Alphabet to use for the one-hot encoding.
    :param sparse: (bool)
        Whether to return a sparse matrix or a numpy array.
    :return:
        Sparse one-hot encoding of the provided sequence
    """

    alpha = len(alphabet)

    # First, convert sequence to list of character indices
    char_to_ix_dict = dict([(c, i) for i, c in enumerate(alphabet)])
    ixs = [char_to_ix_dict[c] for c in seq]

    # Create list of indices for ones, one one for each part
    parts = ohe_spec.split('+')
    num_parts = len(parts)
    offset = 0
    ixs = np.zeros(shape=num_parts, dtype=np.int64)
    for part_num, part in enumerate(parts):
        if part == '.':
            relative_ix = 0
            m = 1
        else:
            poss = [np.int64(pos_str) for pos_str in part.split('x')]
            num_poss = len(poss)
            chars = [seq[pos] for pos in poss]
            relative_ix = sum(
                [(alpha ** (num_poss - 1 - i)) * char_to_ix_dict[c] for i, c in
                 enumerate(chars)])
            m = alpha ** num_poss
        ixs[part_num] = offset + relative_ix
        offset += m

    # Get dimenion of matrix
    M = offset

    # Create sparse column matrix
    data = num_parts * [1]
    i_vals = ixs
    j_vals = num_parts * [0]

    # pdb.set_trace()
    x = sp.coo_array((data, (i_vals, j_vals)), shape=(M, 1)).tocsc()

    return x


# def _ohe_spec_to_BU(ohe_spec_str, alpha=4, compute_inv=False):
#     """
#     :param ohe_spec_str (str): OHE specification string.
#     :param alpha (int >= 2): Size of alphabet.
#     :param compute_inv (bool): Whether to also compute inverses;
#         increase computation time .
#     :return: UB (sparse matrix): Matrix such that BU @ x to x_factored.
#     """
#
#     # Split ohe_spec into parts
#     parts = ohe_spec_str.split('+')
#
#     # Get maximum order
#     max_order = np.max([len(part.split('x')) for part in parts])
#
#     # Get single-position T blocks
#     U_ohe, U_ohe_inv = get_single_position_T_and_T_inv(alpha=alpha)
#     if not compute_inv:
#        U_ohe_inv = sp.csr_matrix(np.zeros(shape=(alpha-1,alpha-1)))
#
#     # Build blocks for order up to maximum order
#     order_to_block_dict = {}
#     BU_triv = sp.csr_array([[1]])
#     BU_part = BU_triv
#     BU_part_inv = BU_triv
#     for k in range(max_order + 1):
#         order_to_block_dict[k] = (BU_part.copy(), BU_part_inv.copy())
#
#         m = alpha ** k
#         i_vals = list(range(m * alpha))
#         j_vals = [alpha * i for i in range(m)] + [
#             i - m + 1 + (i - m) // (alpha - 1) for i in range(m, m * alpha)]
#         data = m * alpha * [1]
#         new_B = sp.coo_array((data, (i_vals, j_vals)),
#                              shape=(alpha * m, alpha * m)).tocsr()
#         # pdb.set_trace()
#         BU_part = new_B @ sp.kron(BU_part, U_ohe)
#         BU_part_inv = sp.kron(BU_part_inv, U_ohe_inv) @ (new_B.T)
#
#     # Build block matrix
#     diag_mats = []
#     diag_mats_inv = []
#     for part in parts:
#         if part == '.':
#             order = 0
#         else:
#             order = len(part.split('x'))
#         BU_part, BU_part_inv = order_to_block_dict[order]
#         diag_mats.append(BU_part)
#         diag_mats_inv.append(BU_part_inv)
#
#     BU = sp.block_diag(diag_mats, format='csr')
#     if compute_inv:
#         BU_inv = sp.block_diag(diag_mats_inv, format='csr')
#     else:
#         BU_inv = None
#
#     return BU, BU_inv


def _ohe_spec_to_T_decom(ohe_spec_str, alpha=4, compute_inv=False):
    """
    :param ohe_spec_str (str): OHE specification string.
    :param alpha (int >= 2): Size of alphabet.
    :param compute_inv (bool): Whether to also compute inverses;
        increase computation time .
    :return: T_decom (sparse matrix)
    """

    # Split ohe_spec into parts
    parts = ohe_spec_str.split('+')

    # Get maximum order
    max_order = np.max([len(part.split('x')) for part in parts])

    # Get single-position T blocks
    T_ohe, T_ohe_inv = get_single_position_T_and_T_inv(alpha=alpha)
    if not compute_inv:
       T_ohe_inv = sp.csr_matrix(np.zeros(shape=(alpha-1,alpha-1)))

    # Build blocks for order up to maximum order
    order_to_block_dict = {}
    T_decom_part = sp.csr_array([[1]])
    T_decom_part_inv = sp.csr_array([[1]])
    order_to_block_dict[0] = (T_decom_part.copy(), T_decom_part_inv.copy())
    for k in range(1, max_order + 1):
        m = alpha ** (k - 1)
        i_vals = list(range(m * alpha))
        j_vals = [alpha * i for i in range(m)] + \
                 [i - m + 1 + (i - m) // (alpha - 1) for i in
                  range(m, m * alpha)]
        data = m * alpha * [1]
        T_perm = sp.coo_array((data, (i_vals, j_vals)),
                             shape=(alpha * m, alpha * m)).tocsr()
        T_decom_part = T_perm @ sp.kron(T_decom_part, T_ohe)
        T_decom_part_inv = sp.kron(T_decom_part_inv, T_ohe_inv) @ (T_perm.T)
        order_to_block_dict[k] = (T_decom_part.copy(), T_decom_part_inv.copy())

    # Build block matrix
    diag_mats = []
    diag_mats_inv = []
    for part in parts:
        if part == '.':
            order = 0
        else:
            order = len(part.split('x'))
        T_decom_part, T_decom_part_inv = order_to_block_dict[order]
        diag_mats.append(T_decom_part)
        diag_mats_inv.append(T_decom_part_inv)

    T_decom = sp.block_diag(diag_mats, format='csr')
    if compute_inv:
        T_decom_inv = sp.block_diag(diag_mats_inv, format='csr')
    else:
        T_decom_inv = None

    return T_decom, T_decom_inv

#
# def _ohe_spec_to_T_decom1(ohe_spec_str, alpha=4, compute_inv=False):
#     """
#     :param ohe_spec_str (str): OHE specification string.
#     :param alpha (int >= 2): Size of alphabet.
#     :param compute_inv (bool): Whether to also compute inverses;
#         increase computation time .
#     :return: T_decom1 (sparse matrix)
#     """
#
#     # Split ohe_spec into parts
#     parts = ohe_spec_str.split('+')
#
#     # Get maximum order
#     max_order = np.max([len(part.split('x')) for part in parts])
#
#     # Get single-position T blocks
#     T_ohe, T_ohe_inv = get_single_position_T_and_T_inv(alpha=alpha)
#     if not compute_inv:
#        T_ohe_inv = sp.csr_matrix(np.zeros(shape=(alpha-1,alpha-1)))
#
#     # Build blocks for order up to maximum order
#     order_to_block_dict = {}
#     T_decom1_part = sp.csr_array([[1]])
#     T_decom1_part_inv = sp.csr_array([[1]])
#     order_to_block_dict[0] = (T_decom1_part.copy(), T_decom1_part_inv.copy())
#     for k in range(1, max_order + 1):
#         T_decom1_part = sp.kron(T_decom1_part, T_ohe)
#         T_decom1_part_inv = sp.kron(T_decom1_part_inv, T_ohe_inv)
#         order_to_block_dict[k] = (T_decom1_part.copy(), T_decom1_part_inv.copy())
#
#     # Build block matrix
#     diag_mats = []
#     diag_mats_inv = []
#     for part in parts:
#         if part == '.':
#             order = 0
#         else:
#             order = len(part.split('x'))
#         T_decom1_part, T_decom1_part_inv = order_to_block_dict[order]
#         diag_mats.append(T_decom1_part)
#         diag_mats_inv.append(T_decom1_part_inv)
#
#     T_decom1 = sp.block_diag(diag_mats, format='csr')
#     if compute_inv:
#         T_decom1_inv = sp.block_diag(diag_mats_inv, format='csr')
#     else:
#         T_decom1_inv = None
#
#     return T_decom1, T_decom1_inv
#

#
# def _ohe_spec_to_T_decom2(ohe_spec_str, alpha=4):
#     """
#     :param ohe_spec_str (str): OHE specification string.
#     :param alpha (int >= 2): Size of alphabet.
#     :return: T_decom2 (sparse matrix)
#     """
#
#     # Split ohe_spec into parts
#     parts = ohe_spec_str.split('+')
#
#     # Get maximum order
#     max_order = np.max([len(part.split('x')) for part in parts])
#
#     # Build blocks for order up to maximum order
#     order_to_block_dict = {}
#     T_decom2_part = sp.csr_array([[1]])
#     T_decom2_part_inv = sp.csr_array([[1]])
#     order_to_block_dict[0] = (T_decom2_part.copy(), T_decom2_part_inv.copy())
#     for k in range(1, max_order + 1):
#         m = alpha ** (k-1)
#         i_vals = list(range(m * alpha))
#         j_vals = [alpha * i for i in range(m)] + \
#             [i - m + 1 + (i - m) // (alpha - 1) for i in range(m, m * alpha)]
#         data = m * alpha * [1]
#         new_B = sp.coo_array((data, (i_vals, j_vals)),
#                              shape=(alpha * m, alpha * m)).tocsr()
#         T_decom2_part = new_B @ sp.kron(T_decom2_part, sp.eye(alpha))
#         T_decom2_part_inv = sp.kron(T_decom2_part_inv, sp.eye(alpha)) @ (new_B.T)
#         order_to_block_dict[k] = (T_decom2_part.copy(), T_decom2_part_inv.copy())
#
#     # Build block matrix
#     diag_mats = []
#     diag_mats_inv = []
#     for part in parts:
#         if part == '.':
#             order = 0
#         else:
#             order = len(part.split('x'))
#         T_decom2_part, T_decom2_part_inv = order_to_block_dict[order]
#         diag_mats.append(T_decom2_part)
#         diag_mats_inv.append(T_decom2_part_inv)
#
#     T_decom2 = sp.block_diag(diag_mats, format='csr')
#     T_decom2_inv = sp.block_diag(diag_mats_inv, format='csr')
#
#     return T_decom2, T_decom2_inv
#



def my_expand(x):
    """
    Expands a list of lists. Simulates product expansion
    """
    if len(x) >= 1:
        a = x[0]
        b = x[1:]
        b_exp = my_expand(b)
        c = [[y]+z for z in b_exp for y in a]
        return c
    else:
        return [x]


### Convert OHE to SIM spec
def ohe_to_sim_spec(ohe_spec_str):
    a = ohe_spec_str.split('+')
    b = [z.split('x') for z in a]
    for i in range(len(b)):
        for j in range(len(b[i])):
            z = b[i][j]
            if z != '.':
                b[i][j] = ['.', z]

    # Recursive expansion
    c = []
    for i, b_el in enumerate(b):
        if isinstance(b_el, str):
            c.append([b_el])
        elif isinstance(b_el, list) and len(b_el) >= 1:
            c.extend(my_expand(b_el))

    # Remove redundant factors of '.'
    sim_spec_list = []
    for x in c:
        y = [z for z in x if z != '.']
        if len(y) == 0:
            y = ['.']
        sim_spec_list.append(y)
    sim_spec_str = '+'.join(['x'.join(z) for z in sim_spec_list])
    return sim_spec_str


# Compute starting positions for each entry in the sim_spec_str
def get_shifts_and_sizes(spec_str, encoding_size):
    """ inputs spec list. outputs a list of (spec, size, shift) """
    spec_list = [x.split('x') for x in spec_str.split('+')]
    specs = []
    shift = 0
    for x in spec_list:
        if len(x)==1 and x[0]=='.':
            size = 1
        else:
            size = encoding_size**len(x)
        specs.append(('x'.join(x),size,shift))
        shift += size
    M = shift
    return specs, M


def _get_thinning_matrix(sim_spec_str, alpha=4):
    # Build zeroing-out matrix
    component_dict = {}

    # Get specs list
    specs, M = get_shifts_and_sizes(sim_spec_str, encoding_size=alpha - 1)

    i_vals = list(range(M))
    j_vals = list(range(M))
    data = M * [1]
    data_inv = M * [1]
    for spec in specs:
        key = spec[0]
        m = spec[1]
        offset = spec[2]
        if key not in component_dict:
            component_dict[key] = (m, offset)
        else:
            m1, offset1 = component_dict[key]
            try:
                assert m1 == m
            except:
                print('m1:', m1)
                print('m:', m)
                pdb.set_trace()

            i_start = offset
            j_start = offset1
            i_vals.extend(list(range(i_start, i_start + m)))
            j_vals.extend(list(range(j_start, j_start + m)))
            data.extend(m * [-1])
            data_inv.extend(m * [1])
    A = sp.coo_array((data, (i_vals, j_vals)), shape=(M, M)).tocsr()
    A_inv = sp.coo_array((data_inv, (i_vals, j_vals)), shape=(M, M)).tocsr()
    return A, A_inv


def get_x_to_test_thinning_matrix(sim_spec_str, alpha=4):
    """
    input: sim_spec_str
    return: x_test
    """
    # Get shifts and sizes
    specs, M = get_shifts_and_sizes(sim_spec_str, encoding_size=alpha-1)

    # Get unique labels
    labels = []
    for spec in specs:
        key = spec[0]
        if not key in labels:
            labels.append(key)

    # Create labels dict
    labels_dict = {}
    counter = 1
    for label in labels:
        labels_dict[label] = .1 + .9 * counter / len(labels)
        counter += 1

        # Build x_test
    x_test = np.zeros(M)
    for spec in specs:
        key = spec[0]
        m = spec[1]
        offset = spec[2]
        r = labels_dict[key]
        x_test[offset:offset + m] = r

    return x_test


def seq_to_desired_BUx(seq, sim_spec_str, alphabet='ACGT'):
    '''
    inputs: seq, ohe_spec, alphabet
    returns: x, a one-hot encoding
    '''
    L = len(seq)
    x_components = []
    x_triv = np.array([1])
    char_to_sim_dict = get_char_to_sim_dict(alphabet=alphabet)

    sim_spec_str_parts = sim_spec_str.split('+')
    for part in sim_spec_str_parts:

        # Add in trivial component
        if part == '.':
            x_components.append(x_triv)
        else:
            positions = [int(p) for p in part.split('x')]
            assert len(positions) > 0
            x_irr = x_triv
            while len(positions) > 0:
                pos = positions.pop(-1)
                c = seq[pos]
                x_l = char_to_sim_dict[c]
                x_irr = np.kron(x_l, x_irr)
            x_components.append(x_irr)

    # Create x
    x = np.concatenate(x_components)
    return x


def _get_distilling_matrix(sim_spec_str, alpha=4):
    # Get specs list
    specs, M = get_shifts_and_sizes(sim_spec_str, encoding_size=alpha-1)

    # Lists to hold i and j values
    component_dict = {}
    nonzero_j_vals = []
    zero_j_vals = []
    next_nonzero_j = 0
    next_zero_j = 0
    next_j = 0
    next_i = 0
    beta = 0
    for spec in specs:
        key = spec[0]
        m = spec[1]
        offset = spec[2]
        next_js = list(range(next_j, next_j + m))
        if key not in component_dict:
            component_dict[key] = (m, offset)
            nonzero_j_vals += next_js
            beta += m
        else:
            zero_j_vals += next_js
        next_j += m
    j_vals = nonzero_j_vals + zero_j_vals
    i_vals = list(range(M))

    data = M * [1]
    D = sp.coo_array((data, (i_vals, j_vals)), shape=(M, M)).tocsr()
    D_inv = D.T
    gamma = M - beta
    return D, D_inv, gamma


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    )


def get_ohe_spec_str(L, n_order, n_adj=None):
    """
    Function to get ohe_spec for any order and num adjacent.
    :param L: (int > 0)
        Length of sequence.
    :param n_order: (int >= 0)
        Highest order of interaction.
    :param n_adj: (int >= 0)
        Maximum span of interaction across positions.
    :return: ohe_spec
        String specification of one-hot encoding.
    """

    # If n_adj is not specified, assume use L
    if n_adj is None:
        n_adj = L

    # Create list of all iterables
    its = []
    for i in range(L - n_adj + 1):
        subset = range(i, i + n_adj)
        for m in range(n_order+1):
            p = list(itertools.combinations(subset,m))
            its.extend(p)
        #p = [s for s in powerset(subset) if len(s) <= n_order]
    its = sorted(set(its), key=lambda x: (len(x), *x))
    its = [[f'{i:d}' for i in it] for it in its]
    spec_str = '.' + '+'.join(['x'.join(it) for it in its])

    # TODO: Fix so that this call is not needed
    # Flip parts
    #spec_str = flip_parts_in_spec_str(spec_str)

    return spec_str


def get_char_to_ohe_dict(alphabet):
    # Make sure all characters in alphabet are unique
    assert len(alphabet) == len(set(alphabet))
    alpha = len(alphabet)
    char_to_ohe_dict = {}
    for i, c in enumerate(alphabet):
        x = np.zeros(alpha)
        x[i] = 1
        char_to_ohe_dict[c] = x
    return char_to_ohe_dict


def get_char_to_sim_dict(alphabet):
    # Make sure all characters in alphabet are unique
    assert len(alphabet) == len(set(alphabet))
    alpha = len(alphabet)
    char_to_sim_dict = {}
    for i, c in enumerate(alphabet[:-1]):
        x = np.zeros(alpha - 1)
        x[i] = 1
        char_to_sim_dict[c] = x
    char_to_sim_dict[alphabet[-1]] = -np.ones(alpha - 1)
    return char_to_sim_dict


def get_single_position_T_and_T_inv(alpha=4):
    M_right = -1 * np.ones((alpha - 1, 1), dtype=np.int64)
    M_top = np.ones([1, alpha], dtype=np.int64)
    M_bulk = np.eye(alpha - 1, dtype=np.int64)
    M_bot = np.concatenate([M_bulk, M_right], axis=1)
    T = np.concatenate([M_top, M_bot], axis=0)

    M_bulk = alpha * np.eye(alpha - 1) - 1
    M_bot = -np.ones([1, alpha - 1])
    M_left = np.ones([alpha, 1])
    M_right = np.concatenate([M_bulk, M_bot], axis=0)
    T_inv = (1 / alpha) * np.concatenate([M_left, M_right], axis=1)

    return sp.csr_array(T), sp.csr_array(T_inv)


def compute_T(ohe_spec,
              alpha,
              verbose=True):
    """
    Computes the T matrix, as well as other distillation info
    :param ohe_spec: (str)
        One-hot encoding specification.
    :param alpha: (int >= 2)
        Size of alphabet.
    :param verbose:
        Whether to print updates to computation.
    :param get_other_info: (bool)
        Whether to return info_dict as well
    :return:
        info_dict: (optional) dict containing other results.
    """

    # Define container class for results
    info_dict = {}
    timing_dict = {}
    obj_dict = {}

    # Start timer
    start_time = time.perf_counter()

    # Get factorization matrix
    if verbose:
        print('_ohe_spec_to_T_decom...')
    t0 = time.perf_counter()
    T_decom, T_decom_inv = _ohe_spec_to_T_decom(ohe_spec,
                                                alpha=alpha,
                                                 compute_inv=True)
    timing_dict['_ohe_spec_to_T_decom'] = time.perf_counter() - t0
    obj_dict['T_decom'] = T_decom
    obj_dict['T_decom_inv'] = T_decom_inv

    # # Get factorization matrix
    # if verbose:
    #     print('_ohe_spec_to_T_decom1...')
    # t0 = time.perf_counter()
    # T_decom1, T_decom1_inv = _ohe_spec_to_T_decom1(ohe_spec,
    #                                                alpha=alpha,
    #                                                compute_inv=True)
    # timing_dict['_ohe_spec_to_T_decom1'] = time.perf_counter() - t0
    # obj_dict['T_decom1'] = T_decom1
    # obj_dict['T_decom1_inv'] = T_decom1_inv

    # # Get expansion matrix
    # if verbose:
    #     print('_ohe_spec_to_T_decom2...')
    # t0 = time.perf_counter()
    # T_decom2, T_decom2_inv = _ohe_spec_to_T_decom2(ohe_spec,
    #                                                alpha=alpha)
    # timing_dict['_ohe_spec_to_T_decom2'] = time.perf_counter() - t0
    # obj_dict['T_decom2'] = T_decom2
    # obj_dict['T_decom2_inv'] = T_decom2_inv

    # Get sim_spec
    if verbose:
        print('ohe_to_sim_spec...')
    t0 = time.perf_counter()
    sim_spec = ohe_to_sim_spec(ohe_spec)
    timing_dict['ohe_to_sim_spec'] = time.perf_counter() - t0

    # Get thinning matrix
    if verbose:
        print('_get_thinning_matrix...')
    t0 = time.perf_counter()
    T_thin, T_thin_inv = _get_thinning_matrix(sim_spec, alpha=alpha)
    timing_dict['_get_thinning_matrix'] = time.perf_counter() - t0
    obj_dict['T_thin'] = T_thin
    obj_dict['T_thin_inv'] = T_thin_inv

    # Get sorting matrix
    if verbose:
        print('_get_distilling_matrix...')
    t0 = time.perf_counter()
    T_sort, T_sort_inv, gamma = _get_distilling_matrix(sim_spec, alpha=alpha)
    timing_dict['_get_distilling_matrix'] = time.perf_counter() - t0
    obj_dict['T_sort'] = T_sort
    obj_dict['T_sort_inv'] = T_sort_inv

    # Get gauge basis
    if verbose:
        print('T and T_inv computation...')
    t0 = time.perf_counter()
    T = T_sort @ T_thin @ T_decom
    T_inv = T_decom_inv @ T_thin_inv @ T_sort_inv
    timing_dict["T and T_inv computation"] = time.perf_counter() - t0
    obj_dict['T'] = T
    obj_dict['T_inv'] = T_inv

    # if verbose:
    #     print('T_inv computation...')
    # t0 = time.perf_counter()
    # T_inv = BU_inv @ T_thin_inv @ T_sort_inv
    # timing_dict["T_inv computation"] = time.perf_counter() - t0

    # Compute gauge bassis
    G_basis = T[-gamma:, :].T
    obj_dict['G_basis'] = G_basis

    # Objects of interest:
    if verbose:
        M = T.shape[0]
        for key, val in obj_dict.items():
            size = val.data.nbytes
            pct = 100*size/(M*M)
            print(f"\t{key}: {size:10,d} bytes, {pct:.3f}% dense.")

    # Gather up other info
    info_dict['T'] = T
    info_dict['gamma'] = gamma
    info_dict['M'] = T.shape[0]
    info_dict['alpha'] = alpha
    info_dict['ohe_spec'] = ohe_spec
    info_dict['T_inv'] = T_inv
    info_dict['timing_dict'] = timing_dict
    info_dict['G_basis'] = G_basis
    info_dict['sparse_intermediates'] = obj_dict

    # End timer
    elapsed_time = time.perf_counter() - start_time

    if verbose:
        print(f'alpha: {alpha:10,d}')
        print(f'    M: {M:10,d}')
        print(f'gamma: {gamma:10,d}')

    print(f'Time for computation to complete: {elapsed_time:.3f} sec.')

    # Return info dict
    return info_dict


def test_distillation(seq,
                      ohe_spec_str,
                      alphabet,
                      show_vecs=True,
                      show_annotations=True,
                      show_xticks=False,
                      num_test_seqs=100,
                      figsize=[10, 5]):

    L = len(seq)
    alpha = len(alphabet)
    print(f'ohe_spec_str: {ohe_spec_str}')


    # Get one-hot encoding of x
    x_ohe = seq_to_x_ohe(seq, ohe_spec_str, alphabet=alphabet).todense()
    print(f'x_ohe.shape: {x_ohe.shape}')
    M = len(x_ohe)

    # Get sim_spec
    sim_spec_str = ohe_to_sim_spec(ohe_spec_str)
    print(f'sim_spec_str: {sim_spec_str}')

    # Get vector to test thinning matrix
    x_test = get_x_to_test_thinning_matrix(sim_spec_str, alpha=alpha)
    x_test = np.array(x_test).reshape([M,1])
    print(f'x_test.shape: {x_test.shape}')

    # Compute T
    info_dict = compute_T(ohe_spec=ohe_spec_str,
                          alpha=alpha,
                          verbose=False)
    T = info_dict['T']
    T_inv = info_dict['T_inv']
    sparse_intermediates_dict = info_dict['sparse_intermediates']
    # T_decom1 = sparse_intermediates_dict['T_decom1']
    # T_decom2 = sparse_intermediates_dict['T_decom2']
    T_decom = sparse_intermediates_dict['T_decom']
    T_thin = sparse_intermediates_dict['T_thin']
    T_sort = sparse_intermediates_dict['T_sort']

    for scalar_name in ['M', 'gamma']:
        scalar = info_dict[scalar_name]
        print(f'{scalar_name}: {scalar:,d}')

    M = info_dict['M']
    gamma = info_dict['gamma']

    info_dict['x_ohe'] = x_ohe
    info_dict['x_test'] = x_test
    info_dict['seq'] = seq
    info_dict['alphabet'] = alphabet
    info_dict['alpha'] = alpha
    info_dict['ohe_spec_str'] = ohe_spec_str
    info_dict['sim_spec_str'] = sim_spec_str

    # Test x_dist is as expected
    all_seqs = make_random_seqs(L, num_seqs=num_test_seqs, alphabet=alphabet)
    num_matches = 0
    for seq in all_seqs:
        x_ohe_ = seq_to_x_ohe(seq, ohe_spec_str, alphabet=alphabet).todense()
        desired_BUx = np.array(
            seq_to_desired_BUx(seq, sim_spec_str, alphabet=alphabet),
            dtype=np.int64).reshape((M, 1))
        match = np.all(T_decom @ x_ohe_ == desired_BUx)
        if match:
            num_matches += 1
        else:
            print(
                f'mismatch for {seq}: T_decom @ x_ohe=' \
                f'\n{(T_decom @ x_ohe_).T};' \
                f'\ndesired_BUx=\n{desired_BUx.T}'
            )
    print(f'T_decom @ x_ohe == x_dist for {num_matches}/{len(all_seqs)} seqs')

    # Test that T and T_inv are inverses
    print('T@T_inv == I:', np.allclose((T @ T_inv).todense(), np.eye(M)))

    # Check that gauge basis is perpendicular to ohe-encoded sequences
    #gauge_basis = T[(M - gamma):, :].T
    gauge_basis = info_dict['G_basis']
    if alpha ** L < num_test_seqs:
        seqs = make_all_seqs(L, alphabet=alphabet)
    else:
        seqs = make_random_seqs(L,
                                num_seqs=num_test_seqs,
                                alphabet=alphabet)
    x_ohes = np.hstack(
        [seq_to_x_ohe(seq, ohe_spec_str, alphabet=alphabet).todense() for seq
         in seqs])
    print(
        f'All {gamma * num_test_seqs:,d} dot products between the ' +
        f'{gamma:,d} gauge vectors and the {num_test_seqs} one-hot ' +
        f'encoded sequences are zero: ',
        np.allclose((gauge_basis.T) @ x_ohes, 0))
    print('Unique elements of gauge basis', np.unique(gauge_basis.data))

    # Show vectors if requested
    if show_vecs:

        # Make figure
        fig, axs = plt.subplots(7, 1, figsize=figsize)

        # Compute ohe vlines
        parts = ohe_spec_str.split('+')
        ohe_vlines = []
        offset = 0
        ohe_vlines.append(offset - .5)
        for part in parts:
            if part == '.':
                m = 1
            else:
                num_poss = len(part.split('x'))
                m = alpha ** (num_poss)
            offset += m
            ohe_vlines.append(offset - .5)

        ohe_annotations = []
        for part_num, part in enumerate(parts):
            x = 0.5*(ohe_vlines[part_num] + ohe_vlines[part_num+1])
            y = 0
            ohe_annotations.append([part, x, y])

        # Compute vline xs
        sim_vlines = []
        parts = sim_spec_str.split('+')
        offset = 0
        sim_vlines.append(offset - .5)
        for part in parts:
            if part == '.':
                m = 1
            else:
                num_poss = len(part.split('x'))
                m = (alpha - 1) ** (num_poss)
            offset += m
            sim_vlines.append(offset - .5)

        sim_annotations = []
        for part_num, part in enumerate(parts):
            x = 0.5*(sim_vlines[part_num] + sim_vlines[part_num+1])
            y = 0
            sim_annotations.append([part, x, y])

        # Compute dist vlines and annotations
        dist_vlines = []
        dist_annotations = []
        part_set = set([])
        offset = 0
        dist_vlines.append(offset - .5)
        for i, part in enumerate(parts):
            if not part in part_set:
                part_set.add(part)
                if part == '.':
                    m = 1
                else:
                    num_poss = len(part.split('x'))
                    m = (alpha - 1) ** (num_poss)
                x = offset + 0.5*m - .5
                y = 0
                offset += m
                dist_vlines.append(offset - .5)
                dist_annotations.append([part, x, y])
            else:
                pass;
        if offset < M:
            x = offset + 0.5*gamma - .5
            y = 0
            dist_annotations.append(['zeros', x, y])
            dist_vlines.append(M-.5)

        test_style_dict = {'color':'k',
                           'ha':'center',
                           'va':'center',
                           'rotation':90}

        def show_vec(ax,
                     vec,
                     title,
                     annotations,
                     vlines,
                     show_xticks=show_xticks,
                     show_annotations=show_annotations):

            # Show x_ohe
            ax.matshow(np.asmatrix(vec).T, vmin=-1, vmax=1, cmap='jet')
            ax.set_yticks([])
            ax.set_title(title)
            ax.set_aspect('auto')
            if show_xticks:
                ax.set_xticks(range(M))
                ax.set_xticklabels([])
            else:
                ax.set_xticks([])
            for x in vlines:
                ax.axvline(x, color='w')
            if show_annotations:
                for part, x, y in annotations:
                    ax.text(s=part, x=x, y=y, **test_style_dict)


        # Show x_ohe
        show_vec(ax=axs[0],
                 vec=x_ohe,
                 title='x_ohe',
                 annotations=ohe_annotations,
                 vlines=ohe_vlines,
                 show_xticks=show_xticks,
                 show_annotations=show_annotations)

        show_vec(ax=axs[1],
                 vec=T_decom @ x_ohe,
                 title='T_decom @ x_ohe',
                 annotations=sim_annotations,
                 vlines=sim_vlines,
                 show_xticks=show_xticks,
                 show_annotations=show_annotations)

        show_vec(ax=axs[2],
                 vec=T_thin @ T_decom @ x_ohe,
                 title='T_thin @ T_decom @ x_ohe',
                 annotations=sim_annotations,
                 vlines=sim_vlines,
                 show_xticks=show_xticks,
                 show_annotations=show_annotations)

        show_vec(ax=axs[3],
                 vec=T_sort @ T_thin @ T_decom @ x_ohe,
                 title='T_sort @ T_thin @ T_decom @ x_ohe',
                 annotations=dist_annotations,
                 vlines=dist_vlines,
                 show_xticks=show_xticks,
                 show_annotations=show_annotations)

        show_vec(ax=axs[4],
                 vec=x_test,
                 title='x_test',
                 annotations=sim_annotations,
                 vlines=sim_vlines,
                 show_xticks=show_xticks,
                 show_annotations=show_annotations)

        show_vec(ax=axs[5],
                 vec=T_thin @ x_test,
                 title='T_thin @ x_test',
                 annotations=sim_annotations,
                 vlines=sim_vlines,
                 show_xticks=show_xticks,
                 show_annotations=show_annotations)

        show_vec(ax=axs[6],
                 vec=T_sort @ T_thin @ x_test,
                 title='T_sort @ T_thin @ x_test',
                 annotations=dist_annotations,
                 vlines=dist_vlines,
                 show_xticks=show_xticks,
                 show_annotations=show_annotations)

        fig.tight_layout()

        info_dict['fig'] = fig
        info_dict['axs'] = axs

    return info_dict


