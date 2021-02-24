from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K


class QHAdam(Optimizer):
    def __init__(self, lr=0.001, beta_1=0.999, beta_2=0.999, v1=0.7, v2=1., epsilon=1e-3, **kwargs):
        super(QHAdam, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.v1 = K.variable(v1, name='v1')
            self.v2 = K.variable(v2, name='v2')
        self.epsilon = epsilon

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        t = K.cast(self.iterations, K.floatx()) + 1

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            m_t_adj = m_t / (1. - K.pow(self.beta_1, t))
            v_t_adj = v_t / (1. - K.pow(self.beta_2, t))

            a = (1. - self.v1) * g + self.v1 * (m_t_adj)
            b = K.sqrt((1. - self.v2) * K.square(g) + self.v2 * v_t_adj) + self.epsilon

            p_t = p - lr * a / b

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                raise NotImplementedError
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'v1': float(K.get_value(self.v1)),
                  'v2': float(K.get_value(self.v2)),
                  'epsilon': self.epsilon}
        base_config = super(QHAdam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))