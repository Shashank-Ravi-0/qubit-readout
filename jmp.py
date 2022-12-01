import jax.tools.colab_tpu
import jax.numpy as jnp
from jax import jit, grad
#jax.tools.colab_tpu.setup_tpu()
import math

# Order is System 1 = qubit/ atom ; System 2 = cavity
class Params:
    def __init__(self, q_ndim, c_ndim, g, alpha, K, w_q, w_c, w_d, A_d):
        self.q_ndim = q_ndim
        self.c_ndim = c_ndim
        self.ndim = self.q_ndim*self.c_ndim
        self.g = g
        self.alpha = alpha
        self.K = K # kappa
        self.w_q = w_q
        self.w_c = w_c
        self.w_d = w_d
        self.A_d = A_d

class Operators:
    def __init__(self, params):
        self.params = params
        self.q = self.q_gen()
        self.q_d = self.q_d_gen()
        self.a = self.a_gen()
        self.a_d = self.a_d_gen()
        self.I_q = self.I_q_gen()
        self.I_c = self.I_c_gen()
        self.s_x = self.s_x_gen()
        self.s_y = self.s_y_gen()
        self.s_z = self.s_z_gen()
        self.x = self.x_gen()
        self.p = self.p_gen()
        self.n = self.n_gen()

    def q_gen(self):
        q = jnp.zeros((self.params.q_ndim, self.params.q_ndim), dtype=float)
        for i in range(self.params.q_ndim - 1):
            q = q.at[i, i+1].set(jnp.sqrt(i+1))
        return q

    def q_d_gen(self):
        q_d = jnp.zeros((self.params.q_ndim, self.params.q_ndim), dtype=float)
        for i in range(self.params.q_ndim - 1):
            q_d = q_d.at[i+1, i].set(jnp.sqrt(i+1))
        return q_d
        
    def a_gen(self):
        a = jnp.zeros((self.params.c_ndim, self.params.c_ndim), dtype=float)
        for i in range(self.params.c_ndim - 1):
            a = a.at[i, i+1].set(jnp.sqrt(i+1))
        return a
    
    def a_d_gen(self):
        a_d = jnp.zeros((self.params.c_ndim, self.params.c_ndim), dtype=float)
        for i in range(self.params.c_ndim - 1):
            a_d = a_d.at[i+1, i].set(jnp.sqrt(i+1))
        return a_d
    
    def x_gen(self):
        return (self.q_d + self.q)/2
        
    def p_gen(self):
        return 1j*(self.q_d - self.q)/2

    def n_gen(self):
        return self.a_d @ self.a       

    def I_q_gen(self):
        return jnp.identity(self.params.q_ndim)
    
    def I_c_gen(self):
        return jnp.identity(self.params.c_ndim)

    def s_x_gen(self):
        return jnp.array([[0,1],[1,0]], dtype=complex)

    def s_y_gen(self):
        return jnp.array([[0,-1j],[1j,0]], dtype=complex)

    def s_z_gen(self):
        return jnp.array([[1,0],[0,-1]], dtype=complex)

class StateVector:
    def __init__(self, params):
        self.params = params

    def fock(self, n):
        lst = [0 for i in range(self.params.c_ndim)]
        lst[n] = 1
        return jnp.array(lst)
    
    def coherent(self):
        s = [(jnp.exp((-(abs(self.params.alpha))**2)/2) * (self.params.alpha)**n)/ jnp.sqrt(math.factorial(n)) for n in range(self.params.c_ndim)]
        return jnp.array(s)