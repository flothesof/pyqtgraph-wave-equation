import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
import numba as nb


@nb.jit
def wave_equation_fd(u, u_prev, c, dx, dt, nt):
    """A finite difference wave equation stencil."""
    for n in range(1, nt):
        u_new = np.zeros_like(u)
        for i in range(1, u.shape[0] - 1):
            for j in range(1, u.shape[1] - 1):
                u_new[i, j] = 2 * u[i, j] - u_prev[i, j] + c ** 2 * dt ** 2 / dx ** 2 * (
                        u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j])
        u, u_prev = u_new, u
    return u, u_prev


def init_wave_randomly(shape):
    """Creates two random wave fields."""
    u = np.random.normal(size=shape, loc=0., scale=0.05)
    u_prev = np.random.normal(size=shape, loc=0., scale=0.05)
    return u, u_prev


def _make_grid():
    """Creates X, Y simulation grid defined by sim parameters."""
    N, M = simulation_params['shape']
    dx = simulation_params['dx']
    dy = dx
    x = np.arange(0, N) * dx
    y = np.arange(0, M) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')  # indexing='ij' is used to keep the N, M aspect
    return X, Y


def init_centered_wave(shape):
    """Creates a gaussian wave field."""
    X, Y = _make_grid()
    source_loc = X.mean(), Y.mean()
    u = np.exp(-((X - source_loc[0]) ** 2 + (Y - source_loc[1]) ** 2) * 50.)
    u_prev = u * 2
    return u, u_prev


def init_plane_wave(shape, k, point_x0, n_wavelengths=1, maxdist=5., decay=10.):
    """Inits a plane wave defined by wave vector k and point x0.

    u(x, t) = exp(1j * (k * (x - x0) - ωt)) * envelope(x - x0, λ)
    """
    X, Y = _make_grid()
    celerity = simulation_params['c']
    dx = simulation_params['dx']
    dt = simulation_params['dt']

    kx, ky = k
    x0, y0 = point_x0

    magnitude_k = np.sqrt(kx ** 2 + ky ** 2)
    omega = magnitude_k * celerity
    wavelength = 2 * np.pi / magnitude_k
    nx, ny = kx / magnitude_k, ky / magnitude_k  # unit vector

    def make_envelope3(x0, y0):
        """Envelope orthognal to propagation direction, smoothed exponentially around the edges."""
        nnx, nny = -ny, nx
        dist = np.abs(nnx * (X - x0) + nny * (Y - y0))
        env = np.ones_like(dist)
        env[dist >= maxdist] *= np.exp(- (dist[dist >= maxdist] - maxdist) * decay)
        return env

    def make_envelope4(x0, y0):
        """Envelope along propagation direction, as a number of wavelengths."""
        dist = np.abs(kx * (X - x0) + ky * (Y - y0))
        env = np.ones_like(dist)
        env[dist >= np.pi * n_wavelengths] *= np.exp(
            - (dist[dist >= np.pi * n_wavelengths] - np.pi * n_wavelengths) * decay)
        return env

    phase0 = kx * (X - x0) + ky * (Y - y0) - np.pi / 2
    u0 = np.cos(phase0) * make_envelope3(x0, y0) * make_envelope4(x0, y0)
    x1, y1 = x0 + celerity * dt * nx, y0 + celerity * dt * ny
    u1 = np.cos(phase0 - omega * dt) * make_envelope3(x1, y1) * make_envelope4(x1, y1)
    return u1, u0


simulation_params = {'c': 1.,
                     'dx': 0.1,
                     'dt': 0.01,
                     'shape': (601, 301)}

app = pg.mkQApp("pyqtgraph-wave-equation")
win = pg.GraphicsLayoutWidget()
win.show()  # show widget alone in its own window
win.setWindowTitle('pyqtgraph-wave-equation')
view = win.addViewBox()
view.setAspectLocked(True)  # lock the aspect ratio so pixels are always square
img = pg.ImageItem(border='w', colorMap=pg.colormap.get('seismic', source='matplotlib'))
view.addItem(img)
view.setRange(QtCore.QRectF(0, 0, simulation_params['shape'][0], simulation_params['shape'][1]))  # set initial view

# Create starting wave
# u, u_prev = init_wave_randomly(simulation_params['shape'])
# u, u_prev = init_centered_wave(simulation_params['shape'])
# u, u_prev = init_plane_wave(simulation_params['shape'], [1.5, 0.], [30., 15.], n_wavelengths=1, maxdist=10., decay=15.)
# u, u_prev = init_plane_wave(simulation_params['shape'], [0., 1.5], [30., 15.], n_wavelengths=2)

# rotated plane wave case
theta = np.deg2rad(-30.)
cos, sin = np.cos(theta), np.sin(theta)
r = np.array([[cos, -sin], [sin, cos]])
u, u_prev = init_plane_wave(simulation_params['shape'], np.dot(r, [0., 2.5]), [30., 15.], n_wavelengths=5)

X, Y = _make_grid()
print(X.max(), Y.max())

# init image
img.setImage(u)
img.setLevels((-1, 1))

elapsed = 0
timer = QtCore.QTimer()
timer.setSingleShot(True)


def updateData():
    global img, u, u_prev, elapsed

    # update wave
    u_new, u = wave_equation_fd(u, u_prev, simulation_params['c'], simulation_params['dx'], simulation_params['dt'], 2)
    u, u_prev = u_new, u
    # Display the data
    img.setImage(u, autoLevels=False)
    timer.start(1)
    elapsed += 1
    # refresh contrast periodically, in a symmetric way
    if elapsed % 1000 == 0:
        mini, maxi = u.min(), u.max()
        abs_max = max(abs(mini), abs(maxi))
        img.setLevels((-abs_max, abs_max))


timer.timeout.connect(updateData)
updateData()
pg.exec()
