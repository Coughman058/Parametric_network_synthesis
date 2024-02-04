'''
The goal of this module is to receive a network from the coupled modes module, and enable the following:
    1. Calculate the eigenmodes of the network, and evaluate their stability
    2. Plot stability diagrams of the network versus some parameter
'''
import sympy as sp
def mMtxFromRules(rules_list, mMtx, mode_cutoff = 12):
  mMtxGGCStab = mMtx.subs(rules_list)
  #.subs(A, A/g).subs(A02c, A02c/g).subs(A, -A).subs(A02c,-A02c)
  mMtxGGCStabBlock = -1*(mMtxGGCStab)[0:mode_cutoff, 0:mode_cutoff]
  # print_latex(mMtxGGCStabBlock)
  return mMtxGGCStabBlock
def eigenValuesFromRules(rules_list, mMtx):
  mMtxGGCStabBlock = mMtxFromRules(rules_list, mMtx = mMtx)
  print("Finding eigenvalues...")
  evals = mMtxGGCStabBlock.eigenvals()
  eval_list = [sp.simplify(-sp.I*eval) for eval in list(evals.keys())];
  return eval_list
def eigenVectorsFromRules(rules_list, mMtx):
  mMtxGGCStabBlock = mMtxFromRules(rules_list, mMtx = mMtx)
  print("Finding eigenvalues...")
  evecs = mMtxGGCStabBlock.eigenvects()
  return evecs
def display_eigensystem(evecs_result):
  for eval in evecs_result:
    print("----------------------")
    print("Eigenvalue:")
    display(sp.simplify(-sp.I*eval[0]))
    print("Multiplicity: ")
    display(eval[1])
    print("Eigenvector")
    [display(sp.simplify(thing)) for thing in eval[2]]
    print("----------------------")


