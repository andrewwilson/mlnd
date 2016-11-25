# https://www.youtube.com/watch?v=2uQ2BSzDvXs
# https://martin-thoma.com/twiddle/

def twiddle(params, param_deltas, fn, threshold, max_iter=10000, scaling=2.0, limits=None, progress_callback=None):
    """

    :param params: e.g. [0,0,0]
    :param param_deltas: e.g. [1,1,1]
    :param fn: the function under test, should return an error value to be minimised.
    :param threshold: iteration will stop when sum of param_deltas is below this threshold
    :param scaling: the rate at which param_deltas is scaled. defaults to 2.0
    :param limits e.g. [[-1,2],None,[150,300], [None,1.0]]
    :return: params
    """
    p = params
    dp = param_deltas

    lims = limits if limits else [None for i in p]

    scaling = float(scaling)
    print "limits", lims

    count = 0
    best_err = fn(p)
    while sum(dp) > threshold and count < max_iter:
        count += 1
        for i in range(len(p)):
            lim = lims[i]
            mn = lim[0] if lim else None
            mx = lim[1] if lim else None
            orig_p = p[i]
            p[i] = limit(dp[i]+p[i], mn, mx)

            err = fn(p)
            if progress_callback:
                progress_callback(err, p, err <= best_err)

            if err < best_err:  # There was some improvement
                best_err = err
                dp[i] *= scaling
            else:  # There was no improvement
                p[i] = limit(orig_p - dp[i], mn,mx)  # Go into the other direction
                err = fn(p)
                if progress_callback:
                    progress_callback(err, p, err <= best_err)

                if err < best_err:  # There was an improvement
                    best_err = err
                    dp[i] *= scaling
                else:  # There was no improvement
                    p[i] = orig_p
                    # As there was no improvement, the step size in either
                    # direction, the step size might simply be too big.
                    dp[i] /= scaling
    return params, count, best_err

def limit(x, mn, mx):
    upper_clipped = min(x,mx) if mx is not None else x
    lower_clipped = max(upper_clipped,mn) if mn is not None else upper_clipped
    #print "limit of ",x,mn,mx,"is", lower_clipped
    return lower_clipped

if __name__ == '__main__':

    X = []
    Y = []
    def func(p):
        x = p[0]
        res = abs(x-100.013)**1.3 -5

        print "trying: ", x, res
        #X.append(x)
        #Y.append(res)
        return res

    #print( func([199.34425354003906]))

    def on_progress(err, params, best):
        if best:
            print  "Progress:", err, best, params
            X.append(params[0])
            Y.append(err)


    params = [0]
    param_deltas = [1]
    print twiddle(params, param_deltas, func, 0.0001, scaling=10, limits=[[5,110.]], progress_callback=on_progress)

    import matplotlib.pyplot as plt
    plt.plot(X, 'o-')
    plt.show()
