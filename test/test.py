import numpy as np
from scipy import interpolate
from mlinterp import RegularGridInterpolator
import time

def make_data(gridvals):
    gridshape = tuple([len(a) for a in gridvals])
    vals = np.random.uniform(-10,10,size=np.prod(gridshape))
    vals = vals.reshape(gridshape)
    return vals

def make_inputs(gridvals, n):
    inputs = np.empty((len(gridvals),n))
    for i,val in enumerate(gridvals):   
        inputs[i,:] = np.random.uniform(val[0],val[-1],size=inputs.shape[1])
    return inputs

def do_interp1(interp, inputs):
    res = np.empty(inputs.shape[1])
    for i in range(inputs.shape[1]):
        res[i] = interp(inputs[:,i].copy())[0]
    return res

def do_interp2(interp, inputs):
    res = np.empty(inputs.shape[1])
    for i in range(inputs.shape[1]):
        res[i] = interp(inputs[:,i].copy())
    return res

def do_test(gridvals, n):
    vals = make_data(gridvals)
    inputs = make_inputs(gridvals, n)

    t1 = time.time()
    interp1 = interpolate.RegularGridInterpolator(gridvals, vals.copy(), method='linear')
    t2 = time.time()
    res1 = do_interp1(interp1, inputs)

    t3 = time.time()

    interp2 = RegularGridInterpolator(gridvals, vals)
    t4 = time.time()
    res2 = do_interp2(interp2, inputs)
    t5 = time.time()

    t1_init = t2 - t1
    t1_calc = t3 - t2
    t2_init = t4 - t3
    t2_calc = t5 - t4

    fmt = '{:15}'
    tmp = fmt.format('%i'%(len(gridvals))) + fmt.format('%.1e'%(t1_init)) + fmt.format('%.1e'%(t2_init)) + fmt.format('%.3f'%(t1_init/t2_init)) + fmt.format('%.1e'%(t1_calc)) + fmt.format('%.1e'%(t2_calc)) + fmt.format('%.1f'%(t1_calc/t2_calc))
    print(tmp)

    assert np.all(np.isclose(res1, res2, atol=1e-100, rtol=1e-10))

def test():
    np.random.seed(0)

    n = 100

    x1 = np.arange(1.0, 10.0, 2.0)
    x2 = np.arange(2.0, 11.0, 3.0)
    x3 = np.arange(2.0, 6.0, 1.0)
    x4 = np.arange(-10.0, 10.0, 2.0)
    x5 = np.arange(2.0, 11.0, 3.0)
    x6 = np.arange(2.0, 6.0, 1.0)
    x7 = np.arange(1.0, 10.0, 2.0)
    x8 = np.arange(2.0, 11.0, 3.0)
    x9 = np.arange(-20.0, 6.0, 5.0)
    x10 = np.arange(1.0, 10.0, 2.0)
    all_gridvals = (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)

    fmt = '{:15}'
    tmp = fmt.format('ndim')+fmt.format('scipy init')+fmt.format('mlinterp init')+fmt.format('scipy/mlinterp')+fmt.format('scipy calc')+fmt.format('mlinterp calc')+fmt.format('scipy/mlinterp')
    print(tmp)

    for i in range(1,len(all_gridvals)+1):
        gridvals = all_gridvals[:i]
        do_test(gridvals, n)

if __name__ == '__main__':
    test()