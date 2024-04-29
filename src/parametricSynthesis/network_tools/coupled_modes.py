'''
This module should contain all the tools to do the following tasks:
    1. Given m modes and N couplings, construct a coupled mode matrix M with decay matrix K symbolically
    2. Add filter networks of arbitrary length to the coupled mode matrix M and decay matrix K
    3. Allow for arbitrary inputs to mode couplings in unitless parameters, and convert between these and the physical units
    4. Calculate input and transfer admittances of the network
    5. Calculate the scattering matrix of the network
'''
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def define_symbols_in_namespace(ns, num_modes = 10, num_pumps = 10):
    # define all the symbols that I need programtically, these are the betas, kappas, etc.
    mMtxElements = [('b%d%d%s' % (j, k, l), sp.symbols('beta%d%d_%s' % (j, k, l))) for k in range(num_modes) for j in
                    range(num_modes) for l in ['g', 'c', 'r']]
    kMtxElements = [('g%d' % j, sp.symbols('gamma%d' % j, positive=True)) for j in range(num_modes)]
    kMtxIntElements = [('g%di' % j, sp.symbols('gamma%d_i' % j, positive=True)) for j in range(num_modes)]
    kMtxExtElements = [('g%de' % j, sp.symbols('gamma%d_e' % j, positive=True)) for j in range(num_modes)]

    filter_deltas = [('D%d%d' % (j, k), sp.symbols('Delta%d%d' % (j, k))) for k in range(num_modes) for j in
                     range(num_modes)]
    filter_betas = [('b%d%d%d%dr' % (j, k, l, m), sp.symbols('beta%d%d_%d%dr' % (j, k, l, m), positive=True)) for k in
                    range(num_modes) for j in range(num_modes) for l in range(num_modes) for m in range(num_modes)]

    mode_freqs = [('omega%d' % j, sp.symbols('omega%d' % j, real=True)) for j in range(num_modes)]
    sig_freqs = [('omegas%d' % j, sp.symbols('omega%d_s' % j, real=True)) for j in range(num_modes)]
    pump_freqs = [('omegap%d' % j, sp.symbols('omega%d_p' % j, real=True)) for j in range(num_pumps)]

    sig_dets = [('ds%d' % j, sp.symbols('delta%d_s' % j)) for j in range(num_modes)]

    pump_det = [('dp%d' % j, sp.symbols('delta%d_p' % j, real=True)) for j in range(num_modes)]
    diag_mMtx_Elements = [('D%d' % j, sp.symbols('Delta%d' % j)) for j in range(num_modes)]

    misc_symbols = [('gamma_m', sp.symbols('gamma_m', positive=True)),('g', sp.symbols('gamma', positive = True))]

    symbols_list = [mMtxElements, kMtxElements, kMtxIntElements, kMtxExtElements, diag_mMtx_Elements, pump_freqs,
                    sig_freqs, mode_freqs, sig_dets, pump_det, filter_deltas, filter_betas, misc_symbols]
    for symbols in symbols_list: ns.update(symbols)

calc_gamma_m = lambda n, ns: (sp.prod([ns['g%i'%i] for i in range(n)]))**(sp.Rational(1,n))
#generate the empty mMtx:
class Mode_Diag():
    def __init__(self, num_modes, namespace = locals()):
        n = num_modes
        mMtx = np.zeros([2*num_modes, 2*num_modes], dtype = 'object')
        kMtx = np.zeros([2*num_modes, 2*num_modes], dtype = 'object')
        for j in range(2*num_modes):
            if j<n:
                mMtx[j, j] = namespace['D%d'%j]
                kMtx[j, j] = sp.sqrt(namespace['g%de'%j])
            if j>=n:
                mMtx[j, j] = -sp.conjugate(namespace['D%d'%(j-n)])
                # mMtx[j, j] = namespace['D%d'%(j-n)]
                kMtx[j, j] = sp.sqrt(namespace['g%de'%(j-n)])
        self.mMtx = mMtx
        self.kMtx = kMtx

    def gen_dmtx(self):
        return sp.Matrix(self.mMtx)
    def gen_kmtx(self):
        return sp.Matrix(self.kMtx)
append_n_zeroes = lambda array, n: np.appand(array, np.zeros(n).astype(int))

def add_filter_modes(mMtx: sp.Matrix,
                     kMtx: sp.Matrix,
                     m: int,
                     n: int,
                     namespace = locals()):
    """
    to be called after generating all the couplings,
    this will add a filter chain of size n to the mth mode

    What this means is that the external gamma will be set to 0 for all
    modes except the external mode that connects to the outside world

    eg it takes the kappa matrix from this
    [1,0,0]
    [0,1,0]
    [0,0,1]

    to this for n = 1, m = 0
    [1,0,0,0]
    [0,0,0,0]
    [0,0,1,0]
    [0,0,0,1]

    and the diagonal on the M matrix from this:
    [d0, 0, 0]
    [0 ,d1, 0]
    [0 , 0,d2]
    where d0 = i*delta -gamma/2
    to this:
    [d01, b, 0, 0]
    [b ,d00, 0, 0]
    [0 , 0,d1,  0]
    [0 , 0, 0, d2]

    where d00 = i*delta
    and d01 = i*delta -gamma/2
    and b has some ridiculous index like b0001r, which is the link between mode
    0's core (0) node in the network and the first filter mode 01
    """
    mSize = mMtx.shape[0] #this should always be even unless you're working with a reduced block-diagonal
    #we have to add in two separate steps for each filter mode, first the row which must contain the
    #we need to substitute in the new delta for the filter network
    delta_to_subs = namespace['D%d'%m]
    replacement_delta = namespace['D%d%d'%(m,0)]
    # display(delta_to_subs)
    # display(replacement_delta)
    mMtx_edit = mMtx.subs(delta_to_subs, replacement_delta)
    kMtx_edit = kMtx
    # display(mMtx_edit)
    for i in range(n):
      diag_delta = namespace['D%d%d'%(m, i+1)]
      beta = namespace['b%d%d%d%dr'%(m, i, m, i+1)]
      blank = np.zeros(mSize+2*i).astype(int)
      nc_row_to_insert = sp.Matrix([list(blank)])
      nc_col_to_insert = sp.Matrix([list(np.append(blank,0))])
      c_row_to_insert = sp.Matrix([list(np.append(blank, [0]))])
      c_col_to_insert = sp.Matrix([list(np.append(blank, [0,0]))])

      kMtx_edit = kMtx_edit.row_insert(m+1, nc_row_to_insert)
      kMtx_edit = kMtx_edit.col_insert(m+1, nc_col_to_insert.transpose())
      kMtx_edit = kMtx_edit.row_insert(m+i+mSize//2+1+1, c_row_to_insert)
      kMtx_edit = kMtx_edit.col_insert(m+i+mSize//2+1+1, c_col_to_insert.transpose())

      mMtx_edit = mMtx_edit.row_insert(m, nc_row_to_insert)
      mMtx_edit = mMtx_edit.col_insert(m, nc_col_to_insert.transpose())
      mMtx_edit = mMtx_edit.row_insert(m+i+mSize//2+1, c_row_to_insert)
      mMtx_edit = mMtx_edit.col_insert(m+i+mSize//2+1, c_col_to_insert.transpose())
      #add the diagonals into the new matrix, the second index decreases in descending order as you go down the diagonal
      # print("before adding deltas")
      # display(mMtx_edit)
      mMtx_edit[m, m] = diag_delta
      mMtx_edit[m,m+1] = beta
      mMtx_edit[m+1,m] = beta

      mMtx_edit[m+i+mSize//2+1, m+i+mSize//2+1] = diag_delta
      mMtx_edit[m+i+mSize//2+1,m+i+mSize//2+1+1] = beta
      mMtx_edit[m+i+mSize//2+1+1,m+i+mSize//2+1] = beta
      # print("i = ", i)
      # display(mMtx_edit)

    return mMtx_edit, kMtx_edit
def swap_basis_vectors(mMtx, kMtx ,m: list,n: list):
  '''
  generate a simple basis switching matrix that will swap all modes m with all
  modes n in the coupled mode matrix of size s
  '''
  mSize = mMtx.shape[0]
  basis_switch_mtx = sp.eye(mSize)
  for i,j in zip(m, n):
    basis_switch_mtx[i,i] = 0
    basis_switch_mtx[j,j] = 0
    basis_switch_mtx[i,j] = 1
    basis_switch_mtx[j,i] = 1
  # display(basis_switch_mtx)
  return basis_switch_mtx*mMtx*basis_switch_mtx, basis_switch_mtx*kMtx*basis_switch_mtx

def ModeReduction(mode_index_to_elim: int, mMtx: sp.Matrix):
  '''
  This will eliminate a mode with index k from the mMtx
  and return a new one with the elementwise formula below

  mMtx[i,j]' = mMtx[i,j] - mMtx[i,k]*mMtx[k,i]/(mMtx[k,k])
  '''
  k = mode_index_to_elim
  mMtx_rank = mMtx.shape[0]
  mMtx_empty = np.zeros((mMtx_rank, mMtx_rank)).astype(int)
  mMtx_new = sp.Matrix(mMtx_empty)
  for i in range(mMtx_rank):
    for j in range(mMtx_rank):
      if j!= k and i!=k:
        mMtx_new[i, j] = mMtx[i,j] - mMtx[i,k]*mMtx[k,j]/(mMtx[k,k])
  mMtx_new.row_del(k)
  mMtx_new.col_del(k)
  return mMtx_new

#add the couplings in by creating a class that embeds the coupling structure within it
#each coupling type has 4 matrix elements except for degenerate gain, which has 2

class Gain:
    def __init__(self, j, k, num_modes = 3, namespace = locals()):
        n = num_modes
        s = namespace['b%d%dg'%(j, k)]
        cpl_mtx = np.zeros([2*num_modes, 2*num_modes], dtype = 'object')

        if k == j:
            #degenerate gain
            cpl_mtx[j, n+k] = s
            cpl_mtx[n+k, j] = -sp.conjugate(s)
        else:
            cpl_mtx[j, n+k] = s
            cpl_mtx[n+k, j] = -sp.conjugate(s)

            cpl_mtx[k, n+j] = s
            cpl_mtx[n+j, k] = -sp.conjugate(s)

        self.cpl_mtx = cpl_mtx

    def gen_mtx(self):
        return sp.Matrix(self.cpl_mtx)
class Conv:
    def __init__(self, j, k, num_modes = 3, namespace = locals()):
        n = num_modes
        s = namespace['b%d%dc'%(j, k)]
        cpl_mtx = np.zeros([2*num_modes, 2*num_modes], dtype = 'object')
        if j == k:
            raise Exception("Degenerate Conversion Not Possible")
        cpl_mtx[j, k] = s
        cpl_mtx[k, j] = sp.conjugate(s)
        cpl_mtx[n+j, n+k] = -sp.conjugate(s)
        cpl_mtx[n+k, n+j] = -s

        self.cpl_mtx = cpl_mtx

    def gen_mtx(self):
        return sp.Matrix(self.cpl_mtx)

class Res:
    def __init__(self, j, k, num_modes = 3, namespace = locals()):
        n = num_modes
        s = namespace['b%d%dr'%(j, k)]
        cpl_mtx = np.zeros([2*num_modes, 2*num_modes], dtype = 'object')
        if j == k:
            raise Exception("Degenerate Resonant Coupling Not Possible")
        cpl_mtx[j, k] = s
        cpl_mtx[k, j] = s
        cpl_mtx[n+j, n+k] = -s
        cpl_mtx[n+k, n+j] = -s

        self.cpl_mtx = cpl_mtx

    def gen_mtx(self):
        return sp.Matrix(self.cpl_mtx)

#tools for calculating scattering

def calculate_impedance_for_config(mMtxN,
                                   kMtxN,
                                   configs,
                                   pump_det_symbol,
                                   signal_det_symbol,
                                   yrange=None, signal_det_range=1,
                                   pump_det=np.linspace(0, 0.5, 3),
                                   fig=None, plot_indices: list = None, ns=locals()):
    pump_det = np.array(pump_det)
    num_modes = mMtxN.shape[0] // 2
    if fig == None:
        if plot_indices == None:
            fig, axs = plt.subplots(ncols=2 * num_modes, figsize=np.array([12, 4]), sharey=True)
            [ax.grid() for ax in axs.flatten()]
            if yrange is not None:
                [ax.set_ylim(yrange) for ax in axs.flatten()]
        else:
            Nplots = np.shape(plot_indices)[0]
            fig, axs = plt.subplots(nrows = 2, ncols=Nplots, figsize=np.array([4 * Nplots, 4]), sharey=True, sharex = True)
            [ax.grid() for ax in axs.flatten()]
            if yrange is not None:
                [ax.set_ylim(yrange) for ax in axs.flatten()]
    else:
        axs = np.array(fig.get_axes()).reshape(-1, int(np.sqrt(np.size(fig.get_axes()))))

    signal_det = np.linspace(-signal_det_range / 2, signal_det_range / 2, 100)

    a = np.shape(configs)[0]
    b = mMtxN.shape[0]
    c = pump_det.size
    d = signal_det.size
    Yin = np.zeros((a, b, c, d)).astype(complex)

    for i, config in enumerate(configs):
        mMtxNFunc = sp.lambdify([pump_det_symbol, signal_det_symbol], mMtxN.subs(config))
        mMtxInv = mMtxN.subs(config).inv()
        kMtxNFunc = sp.lambdify([pump_det_symbol, signal_det_symbol], kMtxN.subs(config))
        gamma_m_val = calc_gamma_m(3, ns).subs(config)
        YinFuncs = []
        for j in range(2 * num_modes):
            mMtxInvComp = sp.simplify(mMtxInv[j, j])
            mMtxInvComp_re = sp.re(mMtxInvComp)
            mMtxInvComp_im = sp.im(mMtxInvComp)
            # display(sp.simplify(mMtxInvComp))
            mMtxInvCompFunc_re = sp.lambdify([pump_det_symbol, signal_det_symbol], mMtxInvComp_re)
            mMtxInvCompFunc_im = sp.lambdify([pump_det_symbol, signal_det_symbol], mMtxInvComp_im)
            mMtxInvCompVals = mMtxInvCompFunc_re(pump_det, signal_det) + 1j * mMtxInvCompFunc_im(pump_det, signal_det)
            YinFunc = lambda pump_det, signal_det: (
                        -0.02 * (2j * gamma_m_val / kMtxN[j, j].subs(config) * 1 / mMtxInvCompVals + 1))
            for k, pd in enumerate(pump_det):
                Yin[i, j, k, :] = YinFunc(pd * np.ones_like(signal_det), signal_det).astype(complex)

        if plot_indices == None:
            for l, j in enumerate(range(b)):  # element of [Yin] / mode to plot
                for k, pd in enumerate(pump_det):
                    axs[l].plot(signal_det, Yin[i, j, k, :].real, label='Real')
                    axs[l].plot(signal_det, Yin[i, j, k, :].imag, label='Imag')
                    axs[l].set_title("Yin_%d" % j)
                    axs[l].legend()

        else:
            for l, j in enumerate(plot_indices):  # element of [Yin] / mode to plot
                for k, pd in enumerate(pump_det):
                    axs[0,l].plot(signal_det, Yin[i, j, k, :].real, label='Real')
                    axs[1,l].plot(signal_det, Yin[i, j, k, :].imag, label='Imag')
                    axs[0,l].set_title("Yin_%d, real" % j)
                    axs[1, l].set_title("Yin_%d, imag" % j)
                    # axs[0,l].legend()

    fig.tight_layout()
    return fig, Yin
def calculate_scattering_for_config(mMtxN,
                                    kMtxN,
                                    configs,
                                    pump_det_symbol,
                                    signal_det_symbol,
                                    yrange = None, signal_det_range = 1,
                                    pump_det = np.linspace(0,0.5,3),
                                    fig = None, plot_pairs = None):
  num_modes = mMtxN.shape[0]//2
  if fig == None:
    if plot_pairs == None:
      fig, axs = plt.subplots(nrows = 2*num_modes, ncols = 2*num_modes, figsize = np.array([24,16])*0.5, sharey = True)
      [ax.grid() for ax in axs.flatten()]
      if yrange is not None:
        [ax.set_ylim(yrange) for ax in axs.flatten()]
    else:
      Nplots = np.shape(plot_pairs)[0]
      fig, axs = plt.subplots(ncols = Nplots, figsize = np.array([4*Nplots,4]), sharey = True)
      [ax.grid() for ax in axs.flatten()]
      if yrange is not None:
        [ax.set_ylim(yrange) for ax in axs.flatten()]
  else:
    axs = np.array(fig.get_axes()).reshape(-1, int(np.sqrt(np.size(fig.get_axes()))))

  for config in configs:
    mMtxNFunc = sp.lambdify([pump_det_symbol, signal_det_symbol], mMtxN.subs(config))
    kMtxNFunc = sp.lambdify([pump_det_symbol, signal_det_symbol], kMtxN.subs(config))
    sMtxNFunc = lambda pump_det, sig_det: 1j*np.matmul(
        np.matmul(kMtxNFunc(pump_det,sig_det),
                  np.linalg.inv(mMtxNFunc(pump_det,sig_det))),
        kMtxNFunc(pump_det,sig_det)
        )-np.identity(mMtxN.shape[0])
    # display(mMtxNFunc(0,0))
    signal_det = np.linspace(-signal_det_range/2,signal_det_range/2,100)
    sMtxs = np.array([np.array([sMtxNFunc(pd,d) for d in signal_det]) for pd in pump_det])
    if plot_pairs == None:
      for i in range(len(sMtxs)):
          for j in range(2*num_modes):
              for k in range(2*num_modes):
                axs[j,k].plot(signal_det, 20*np.log10(np.abs(sMtxs[i,:,j,k])), label = "$\delta_p = $"+str(pump_det[i])+" $\kappa$")
                axs[l].set_title("S%d%d" % (j, k))
    else:
      for i in range(len(sMtxs)):
        for l, pair in enumerate(plot_pairs):
          j,k = pair
          axs[l].plot(signal_det, 20*np.log10(np.abs(sMtxs[i,:,j,k])))
          axs[l].set_title("S%d%d"%(j,k))

  fig.tight_layout()
  return fig



#
# def ModeReduction(mode_index_to_elim: int, mMtx: sp.Matrix):
#   '''
#   This will eliminate a mode with index k from the mMtx
#   and return a new one with the elementwise formula below
#
#   mMtx[i,j]' = mMtx[i,j] - mMtx[i,k]*mMtx[k,i]/(mMtx[k,k])
#   '''
#   k = mode_index_to_elim
#   mMtx_rank = mMtx.shape[0]
#   mMtx_empty = np.zeros((mMtx_rank, mMtx_rank)).astype(int)
#   mMtx_new = sp.Matrix(mMtx_empty)
#   for i in range(mMtx_rank):
#     for j in range(mMtx_rank):
#       if j!= k and i!=k:
#         mMtx_new[i, j] = mMtx[i,j] - mMtx[i,k]*mMtx[k,j]/(mMtx[k,k])
#   mMtx_new.row_del(k)
#   mMtx_new.col_del(k)
#   return mMtx_new
#
# calc_gamma_m = lambda n, ns: (sp.prod([ns['g%i'%i] for i in range(n)]))**(sp.Rational(1,n))
# gs_to_betas = lambda gs, dw, g0: dw/(2*g0*np.sqrt(gs*np.roll(gs, -1)))[1]

if __name__ == "__main__":
    #define the namespace
    ns = locals()
    define_symbols_in_namespace(ns, num_modes = 3, num_pumps = 3)
    #define the mode matrix
    mode_diag = Mode_Diag(3, namespace = ns)
    mMtx = mode_diag.gen_dmtx()
    kMtx = mode_diag.gen_kmtx()
    #add the couplings
    gain01 = Gain(0, 1, num_modes = 3, namespace = ns)
    gain12 = Gain(1, 2, num_modes = 3, namespace = ns)
    conv = Conv(0, 2, num_modes = 3, namespace = ns)
    mMtx = mMtx + gain01.gen_mtx() + conv.gen_mtx() + gain12.gen_mtx()
    mMtx = swap_basis_vectors(mMtx, kMtx, [1], [4])
    #add the filter modes
    breakpoint()