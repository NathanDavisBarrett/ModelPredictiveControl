import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt


def PlotFullModel(model, params):
    TIME = [t for t in model.TIME]
    X = [model.X[t, 0].value for t in model.TIME]
    Y = [model.X[t, 1].value for t in model.TIME]
    Z = [model.X[t, 2].value for t in model.TIME]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xi = params.x0[0]
    yi = params.x0[1]
    zi = params.x0[2]

    xf = params.xf[0]
    yf = params.xf[1]
    zf = params.xf[2]

    ax.set_xlim([min(xi, xf), max(xi, xf)])
    ax.set_ylim([min(yi, yf), max(yi, yf)])
    ax.set_zlim([min(zi, zf), max(zi, zf)])

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    ax.plot(X, Y, Z, "b-")
    plt.show()
