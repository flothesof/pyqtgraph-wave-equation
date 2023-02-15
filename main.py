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
    u = np.random.normal(size=shape, loc=1024, scale=64)
    u_prev = np.random.normal(size=shape, loc=1024, scale=64)
    return u, u_prev


def init_centered_wave(shape):
    """Creates a gaussian wave field."""
    N, M = simulation_params['shape']
    dx = simulation_params['dx']
    dy = dx
    x = np.arange(0, N) * dx
    y = np.arange(0, M) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')  # indexing='ij' is used to keep the N, M aspect

    source_loc = x.mean(), y.mean()
    u = np.exp(-((X - source_loc[0]) ** 2 + (Y - source_loc[1]) ** 2) * 50.)
    u_prev = u * 2
    return u, u_prev


simulation_params = {'c': 1.,
                     'dx': 0.1,
                     'dt': 0.01,
                     'shape': (600, 300)}

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
u, u_prev = init_centered_wave(simulation_params['shape'])

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
