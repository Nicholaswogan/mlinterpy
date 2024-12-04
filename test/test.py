import numpy as np
from scipy import interpolate
from mlinterpy import RegularGridInterpolator
from copy import deepcopy
import time

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

def do_test(gridvals, n, vectorized):
    vals = make_data(gridvals)
    inputs = make_inputs(gridvals, n)
    
    t1 = time.time()
    interp1 = interpolate.RegularGridInterpolator(deepcopy(gridvals), vals.copy(), method='linear')
    t2 = time.time()

    if vectorized:
        t3 = time.time()
        res1 = interp1(inputs)
        t4 = time.time()
    else:
        t3 = time.time()
        res1 = do_interp1(interp1, inputs)
        t4 = time.time()

    t5 = time.time()
    interp2 = RegularGridInterpolator(gridvals, vals)
    t6 = time.time()

    if vectorized:
        t7 = time.time()
        res2 = interp2.evaluate_vector(inputs)
        t8 = time.time()
    else:
        t7 = time.time()
        res2 = do_interp2(interp2, inputs)
        t8 = time.time()

    t1_init = np.maximum(t2 - t1,1e-100)
    t1_calc = np.maximum(t4 - t3,1e-100)

    t2_init = np.maximum(t6 - t5,1e-100)
    t2_calc = np.maximum(t8 - t7,1e-100)

    fmt = '{:15}'
    tmp = fmt.format('%i'%(len(gridvals))) + fmt.format('%r'%(vectorized)) + fmt.format('%.1e'%(t1_init)) + fmt.format('%.1e'%(t2_init)) + fmt.format('%.3f'%(t1_init/t2_init)) + fmt.format('%.1e'%(t1_calc)) + fmt.format('%.1e'%(t2_calc)) + fmt.format('%.1f'%(t1_calc/t2_calc))
    print(tmp)

    assert np.allclose(res1, res2, atol=1e-100, rtol=1e-10)

def test():
    np.random.seed(0)

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
    all_gridvals = (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)

    fmt = '{:15}'
    tmp = fmt.format('ndim')+fmt.format('vectorized')+fmt.format('scipy init')+fmt.format('mlinterp init')+fmt.format('scipy/mlinterp')+fmt.format('scipy calc')+fmt.format('mlinterp calc')+fmt.format('scipy/mlinterp')
    print(tmp)

    for i in range(1,len(all_gridvals)+1):
        gridvals = all_gridvals[:i]
        do_test(gridvals, n, False)
        do_test(gridvals, n, True)

if __name__ == '__main__':
    test()