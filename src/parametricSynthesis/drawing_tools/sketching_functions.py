import schemdraw
import schemdraw.elements as elm
import numpy as np

def draw_net_by_type(net, Ftype, l = 3.5):
  Ftypes = [
        'cap_cpld_lumped',
        'tline_cpld_lumped',
        'tline_cpld_l4',
        'tline_cpld_l2',
        'cap_cpld_l4']
  #TODO: add 'ideal' with regular inverters
  Ftype = Ftype.lower()

  if Ftype not in Ftypes:
    raise Exception(f'net not supported, choose between {Ftypes}')

  if Ftype == 'cap_cpld_lumped':
    d = sketch_cap_cpld_lumped(net, l = l)

  elif Ftype == 'tline_cpld_lumped':
    d = sketch_TLINE_cpld_lumped(net, l = l)

  elif Ftype == 'tline_cpld_l4':
    d = sketch_TLINE_cpld_lambda_over_4(net, l = l)

  elif Ftype == 'tline_cpld_l2':
    d = sketch_TLINE_cpld_lambda_over_2(net, l = l)

  elif Ftype == 'cap_cpld_l4':
    d = sketch_cap_cpld_lambda_over_4(net, l = l)

  return d

def sketch_ideal_inverter_net(net, l = 1.5):
    with schemdraw.Drawing() as d:
        d += elm.Ground()
        d += elm.RBox(label="$g_0$").up()
        net_size = net.J.size
        for n in range(net_size+2):
          if n%2 == 0:
            d += elm.RBox(label=f"$g_{n+1}$").right()
            d.push()
          else:
            d += elm.RBox(label=f"$g_{n+1}$").down()
            d += elm.Ground()
            d.pop()
    return d
def sketch_TLINE_cpld_lumped(net, l = 1.5):
  with schemdraw.Drawing() as d:
    d += elm.Ground()
    d += elm.RBox(label="$Z_0$").up()
    net_size = net.J.size-1
    for n in range(net_size+1):
      d += elm.Coax(label = f"$\lambda/4$\n$Z_c$ = {np.round(1/net.J[net_size-n]*net.tline_inv_Z_corr_factor, 1)} $\Omega$").scale(1).right()
      d.push()
      d += elm.Inductor2().down().label(f"$L =$ {np.round(net.L[net_size-n]*1e12, 1)} pH", loc = 'bottom')
      d += elm.Line().right().length(l/2)
      d.pop()
      d += elm.Line().right().length(l)
      d.push()
      d += elm.Capacitor().down().label(f"\n\n$C =$ {np.round(net.Cu[net_size-n]*1e12, 3)} pF", loc = 'top')
      d += elm.Line().left().length(l/2)
      d += elm.Ground()
      d.pop()
  return d


def sketch_cap_cpld_lumped(net, l = 1.5):
  with schemdraw.Drawing() as d:
    d += elm.Ground()
    d += elm.RBox(label="$Z_0$").up()
    net_size = net.J.size-1
    for n in range(net_size+1):
      d += elm.Capacitor(label = f"C = {np.round(net.CC[net_size-n]*1e12*net.tline_inv_Z_corr_factor, 3)} pF").scale(1).right().length(l/2)
      d.push()
      d += elm.Inductor2().down().label(f"$L =$ {np.round(net.L[net_size-n]*1e12, 1)} pH", loc = 'bottom')
      d += elm.Line().right().length(l/2)
      d.pop()
      d += elm.Line().right().length(l)
      d.push()
      d += elm.Capacitor().down().label(f"\n\n$C =$ {np.round(net.C[net_size-n]*1e12, 3)} pF", loc = 'top')
      d += elm.Line().left().length(l/2)
      d += elm.Ground()
      d.pop()
  return d


def sketch_TLINE_cpld_lambda_over_2(net, l = 1.5):
  with schemdraw.Drawing() as d:
    d += elm.Ground()
    d += elm.RBox(label="$Z_0$").up()
    net_size = net.J.size-1
    for n in range(net_size):
      d += elm.Coax(label = f"$\lambda/4$\n$Z_c$={np.round(1/net.J[net_size-n]*net.tline_inv_Z_corr_factor, 1)} $\Omega$").scale(1).right()
      d.push()
      d += elm.Coax().down().label(f"$\lambda/2$\n$Z_c=$ {np.round(net.Z[net_size-n]*np.pi/2, 1)} $\Omega$", loc = 'top')
      d.pop()
    d += elm.Coax(label = f"$\lambda/4$ \n $Z_c$ = {np.round(1/net.J[0]*net.tline_inv_Z_corr_factor, 1)} $\Omega$").scale(1).right()
    d.push()
    d += elm.Inductor2().down().label(f"$L =$ {np.round(net.L[0]*1e12, 1)} pH", loc = 'bottom')
    d += elm.Line().right().length(l/2)
    d.pop()
    d += elm.Line().right().length(l)
    d.push()
    d += elm.Capacitor().down().label(f"\n\n$C =$ {np.round(net.Cu[0]*1e12, 3)} pF", loc = 'top')
    d += elm.Line().left().length(l/2)
    d += elm.Ground()
    d.pop()
  return d


def sketch_TLINE_cpld_lambda_over_4(net, l = 1.5):
  with schemdraw.Drawing() as d:
    d += elm.Ground()
    d += elm.RBox(label="$Z_0$").up()
    net_size = net.J.size-1
    for n in range(net_size):
      d += elm.Coax(label = f"$\lambda/4$\n$Z_c$={np.round(1/net.J[net_size-n]*net.tline_inv_Z_corr_factor, 1)} $\Omega$").scale(1).right()
      d.push()
      d += elm.Coax().down().label(f"$\lambda/4$\n$Z_c=$ {np.round(net.Z[net_size-n]*np.pi/4, 1)} $\Omega$", loc = 'top')
      d += elm.Ground()
      d.pop()
    d += elm.Coax(label = f"$\lambda/4$ \n $Z_c$ = {np.round(1/net.J[0]*net.tline_inv_Z_corr_factor, 1)} $\Omega$").scale(1).right()
    d.push()
    d += elm.Inductor2().down().label(f"$L =$ {np.round(net.L[0]*1e12, 1)} pH", loc = 'bottom')
    d += elm.Line().right().length(l/2)
    d.pop()
    d += elm.Line().right().length(l)
    d.push()
    d += elm.Capacitor().down().label(f"\n\n$C =$ {np.round(net.Cu[0]*1e12, 3)} pF", loc = 'top')
    d += elm.Line().left().length(l/2)
    d += elm.Ground()
    d.pop()
  return d

def sketch_cap_cpld_lambda_over_4(net, l = 1.5):
  with schemdraw.Drawing() as d:
    d += elm.Ground()
    d += elm.RBox(label="$Z_0$").up()
    net_size = net.J.size-1
    for n in range(net_size):
      d += elm.Capacitor(
        label=f"C = {np.round(net.CC[net_size - n] * 1e12 * net.tline_inv_Z_corr_factor, 3)} pF").scale(
        1).right().length(l)
      d.push()
      d += elm.Coax().down().label(r"$\theta=$" + f"{np.round(net.theta[net_size-n]*360/2/np.pi, 2)} \n$Z_c=$ {np.round(net.Z[net_size-n]*np.pi/4, 1)} $\Omega$", loc = 'top')
      d += elm.Ground()
      d.pop()
    d += elm.Capacitor(
      label=f"Cc = {np.round(net.CC[0] * 1e12 * net.tline_inv_Z_corr_factor, 3)} pF").scale(
      1).right().length(l)
    d.push()
    d += elm.Inductor2().down().label(f"$L =$ {np.round(net.L[0]*1e12, 1)} pH", loc = 'bottom')
    d += elm.Line().right().length(l/2)
    d.pop()
    d += elm.Line().right().length(l)
    d.push()
    d += elm.Capacitor().down().label(f"\n\n$C =$ {np.round(net.C[0]*1e12, 3)} pF", loc = 'top')
    d += elm.Line().left().length(l/2)
    d += elm.Ground()
    d.pop()
  return d