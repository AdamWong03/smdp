"""Time-origin averaged molecular MSD, with explicit fitting conventions."""
import numpy as np


def time_averaged_msd(coordinates):
    x=np.asarray(coordinates,dtype=float)
    x=x-x[:1]
    n,m,_=x.shape
    squared=np.sum(x*x,axis=(1,2))
    prefix=np.concatenate([[0.],np.cumsum(squared)])
    fft=np.fft.rfft(x,n=2*n,axis=0)
    power=np.sum(fft.conj()*fft,axis=(1,2))
    corr=np.fft.irfft(power,n=2*n)[:n]
    lag=np.arange(n)
    result=(prefix[n]-prefix[lag]+prefix[n-lag]-2*corr)/((n-lag)*m)
    result[0]=0.
    return result
