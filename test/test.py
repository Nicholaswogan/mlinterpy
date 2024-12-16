import numpy as np
from scipy import interpolate
from mlinterpy import MultiLinearInterpolator
from copy import deepcopy
import timeit

def make_data(gridvals):
    gridshape = tuple([len(a) for a in gridvals])
    vals = np.random.uniform(-10,10,size=np.prod(gridshape))
    vals = vals.reshape(gridshape)
    return vals

def make_inputs(gridvals, n):
    inputs = np.empty((n,len(gridvals)))
    for i,val in enumerate(gridvals):   
        inputs[:,i] = np.random.uniform(val[0],val[-1],size=inputs.shape[0])
    return inputs

def do_interp1(interp, inputs):
    res = np.empty(inputs.shape[0])
    for i in range(inputs.shape[0]):
        res[i] = interp(inputs[i,:])[0]
    return res

def do_interp2(interp, inputs):
    res = np.empty(inputs.shape[0])
    for i in range(inputs.shape[0]):
        res[i] = interp.evaluate(inputs[i,:])
    return res

def timefunc(func):
    timer = timeit.Timer(func)
    number, t = timer.autorange()
    return t/number

def do_test(gridvals, n, vectorized, time):
    np.random.seed(0)

    vals = make_data(gridvals)
    inputs = make_inputs(gridvals, n)
    
    tmp = lambda: interpolate.RegularGridInterpolator(deepcopy(gridvals), vals.copy())
    interp1 = tmp()
    if time:
        t1_init = timefunc(tmp)

    if vectorized:
        tmp = lambda: interp1(inputs)
    else:
        tmp = lambda: do_interp1(interp1, inputs)

    res1 = tmp()
    if time:
        t1_calc = timefunc(tmp)

    tmp = lambda: MultiLinearInterpolator(gridvals, vals)
    interp2 = tmp()
    if time:
        t2_init = timefunc(tmp)

    if vectorized:
        inputs1 = np.ascontiguousarray(inputs.T)
        tmp = lambda: interp2.call(inputs1)
    else:
        tmp = lambda: do_interp2(interp2, inputs)

    res2 = tmp()
    if time:
        t2_calc = timefunc(tmp)

    if time:
        fmt = '{:16}'
        tmp = fmt.format('%i'%(len(gridvals))) + fmt.format('%r'%(vectorized)) + fmt.format('%.1e'%(t1_init)) + fmt.format('%.1e'%(t2_init)) + fmt.format('%.3f'%(t1_init/t2_init)) + fmt.format('%.1e'%(t1_calc)) + fmt.format('%.1e'%(t2_calc)) + fmt.format('%.1f'%(t1_calc/t2_calc))
        print(tmp)

    assert np.allclose(res1, res2, atol=1e-100, rtol=1e-10)

def test():

    time = False
    n = 100

    x1 = np.arange(1.0, 10.0, 2.0)
    x2 = np.arange(2.0, 11.0, 3.0)
    x3 = np.arange(2.0, 6.0, 1.0)
    x4 = np.arange(-1.0, 10.0, 2.0)
    x5 = np.arange(2.0, 11.0, 3.0)
    x6 = np.arange(2.0, 6.0, 1.0)
    x7 = np.arange(1.0, 10.0, 2.0)
    x8 = np.arange(2.0, 11.0, 3.0)
    x9 = np.arange(-20.0, 6.0, 5.0)
    x10 = np.arange(1.0, 10.0, 2.0)
    x11 = np.arange(1.0, 6.0, 2.0)
    all_gridvals = (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)

    if time:
        fmt = '{:16}'
        tmp = fmt.format('ndim')+fmt.format('vectorized')+fmt.format('scipy init')+fmt.format('mlinterpy init')+fmt.format('scipy/mlinterpy')+fmt.format('scipy calc')+fmt.format('mlinterpy calc')+fmt.format('scipy/mlinterpy')
        print(tmp)

    for i in range(1,len(all_gridvals)+1):
        gridvals = all_gridvals[:i]
        do_test(gridvals, n, False, time)
        do_test(gridvals, n, True, time)

if __name__ == '__main__':
    test()