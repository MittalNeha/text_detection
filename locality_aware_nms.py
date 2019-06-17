import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
    g = Polygon(g)
    p = Polygon(p)
    #g = Polygon(g[:8].reshape((4, 2)))
    #p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, vg, p, vp):
    g[:8] = (vg * g[:8] + vp * p[:8])/(vg + vp)
    vp = (vg + vp)/2
    return g, vp


def standard_nms(S, conf, thres):
    #order = np.argsort(S[:, 8])[::-1]
    order = np.argsort(conf)[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep], np.array(conf)[keep]


def nms_locality(polys, conf, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    vS = []
    p = None
    vp = None
    
    #print("poly shape = {}".format(polys.shape))
    for idx, g in enumerate(polys):
        if p is not None and intersection(g, p) > thres:
            vg = conf[idx]
            p, vp = weighted_merge(g, vg, p, vp)
        else:
            if p is not None:
                S.append(p)
                vS.append(vp)
            p = g
            vp = conf[idx]
    if p is not None:
        S.append(p)
        vS.append(vp)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), vS, thres)


if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
