'''Periodic & smooth image decomposition'''
import numpy as np
import skimage

def u2v(u: np.ndarray) -> np.ndarray:
    '''Converts the image `u` into the image `v`

    Parameters
    ----------
    u : np.ndarray
        [M, N] image

    Returns
    -------
    v : np.ndarray
        [M, N] image, zeroed expect for the outermost rows and cols
    '''
    v = np.zeros(u.shape, dtype=u.dtype)

    v[0, :] = u[-1, :] - u[0,  :]
    v[-1,:] = u[0,  :] - u[-1, :]

    v[:,  0] += u[:, -1] - u[:,  0]
    v[:, -1] += u[:,  0] - u[:, -1]
    return v

def v2s(v_hat: np.ndarray) -> np.ndarray:
    '''Computes the maximally smooth component of `u`, `s` from `v`


    s[q, r] = v[q, r] / (2*np.cos( (2*np.pi*q)/M ) + 2*np.cos( (2*np.pi*r)/N ) - 4)

    Parameters
    ----------
    v_hat : np.ndarray
        [M, N] DFT of v
    '''
    M, N = v_hat.shape

    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2*np.cos( np.divide((2*np.pi*q), M) ) + 2*np.cos( np.divide((2*np.pi*r), N) ) - 4)
    s = v_hat / den
    s[0, 0] = 0
    return s

def periodic_smooth_decomp(I: np.ndarray) -> (np.ndarray, np.ndarray):
    '''Performs periodic-smooth image decomposition

    Parameters
    ----------
    I : np.ndarray
        [M, N] image. will be coerced to a float.

    Returns
    -------
    P : np.ndarray
        [M, N] image, float. periodic portion.
    S : np.ndarray
        [M, N] image, float. smooth portion.
    '''
    u = skimage.img_as_float(I)
    v = u2v(u)
    v_fft = np.fft.fftn(v)
    s = v2s(v_fft)
    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f # u = p + s
    return p, s_f

if __name__ == '__main__':
    '''Plot the astronaut with and without P&S decomp'''
    import matplotlib
    matplotlib.rcParams.update({'font.size':30})
    import matplotlib.pyplot as plt
    from skimage.data import astronaut

    def plot_img(I: np.ndarray, title: str=None, out: str=None) -> None:
        '''Plot an image with a nice big figure'''
        plt.figure(None, (10,10));
        plt.imshow(I, cmap='gray');
        if title is not None:
            plt.title(title)
        if out is not None:
            plt.savefig(out)
        return

    def viz_fft(f: np.ndarray, out: str) -> None:
        shift = np.fft.fftshift(f)
        amp = np.log(np.abs(shift) + 1)
        phase = np.angle(shift)
        fig, ax = plt.subplots(1,2,figsize=(16,8))
        ax[0].imshow(amp, cmap='gray')
        ax[0].set_title('Amplitude')
        ax[1].imshow(phase, cmap='gray')
        ax[1].set_title('Phase')
        plt.tight_layout()
        plt.savefig(out)
        return

    Irgb = astronaut()
    Ig = skimage.color.rgb2gray(Irgb)
    Ig = skimage.img_as_float(Ig)

    p, s = periodic_smooth_decomp(Ig)

    fig, ax = plt.subplots(3, 3, figsize=(20,20))

    ylabs = ['u', 'p', 's']
    for i, j in enumerate([Ig, p, s]):
        jf = np.fft.fftn(j)
        ax[i, 0].imshow(j, cmap='gray')
        ax[i, 0].set_ylabel(ylabs[i])
        ax[i, 1].imshow(np.log(np.abs(np.fft.fftshift(jf)) + 1), cmap='gray')
        ax[i, 2].imshow(np.angle(np.fft.fftshift(jf)), cmap='gray')
        for k in range(3):
            ax[i, k].set_xticks([])
            ax[i, k].set_yticks([])

    ax[0,0].set_title('Image')
    ax[0,1].set_title('log(Amplitude+1)')
    ax[0,2].set_title('Phase')
    plt.tight_layout()
    plt.savefig('astronaut_psd.png')
