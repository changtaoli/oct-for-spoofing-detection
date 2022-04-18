from numpy import log, exp, infty, zeros_like, vstack, zeros, errstate, finfo, sqrt, floor, tile, concatenate, \
    arange, meshgrid, ceil, linspace
from scipy.special import logsumexp
from scipy.signal import lfilter
from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.exceptions import ParameterError, ErrorMsgs
from spafe.fbanks.linear_fbanks import linear_filter_banks
from spafe.fbanks.mel_fbanks import mel_filter_banks, inverse_mel_filter_banks
from spafe.utils.cepstral import cms, cmvn, lifter_ceps
from spafe.utils.spectral import dct, power_spectrum, rfft 
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
from librosa import util
import numpy as np


def Deltas(x, width=3):
    hlen = int(floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]


def lfcc(sig,
         fs=16000,
         num_ceps=20,
         pre_emph=0,
         pre_emph_coeff=0.97,
         win_len=0.030,
         win_hop=0.015,
         win_type="hamming",
         nfilts=40,
         nfft=1024,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         normalize=0,
         order_deltas=2):
    """
    Compute the linear-frequency cepstral coefﬁcients (GFCC features) from an audio signal.
    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.
    Returns:
        (array) : 2d array of LFCC features ((num_ceps x 3) x num_frames)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = np.fft.rfft(windows, nfft)
    abs_fft_values = np.abs(fourrier_transform)**2

    #  -> x linear-fbanks
    linear_fbanks_mat = linear_filter_banks(nfilts=nfilts,
                                            nfft=nfft,
                                            fs=fs,
                                            low_freq=low_freq,
                                            high_freq=high_freq,
                                            scale=scale)

    features = np.dot(abs_fft_values, linear_fbanks_mat.T)

    log_features = np.log10(features+2.2204e-16)

    #  -> DCT(.)
    lfccs = dct(log_features, type=dct_type,
                norm='ortho', axis=1)[:, :num_ceps]
    lfccs = lfccs.T

    if order_deltas > 0:
        feats = list()
        feats.append(lfccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = np.vstack(feats)
    # vstack:按照垂直方向
    # hstack:按照水平方向

    return lfccs


def msrcc(sig,
          fs=16000,
          num_ceps=13,
          pre_emph=1,
          pre_emph_coeff=0.97,
          win_len=0.02,
          win_hop=0.01,
          win_type="hamming",
          nfilts=40,
          nfft=512,
          low_freq=None,
          high_freq=None,
          scale="constant",
          gamma=-1 / 7,
          dct_type=2,
          use_energy=False,
          lifter=22,
          normalize=0,
          order_deltas=2):
    """
    Compute the Magnitude-based Spectral Root Cepstral Coefﬁcients (MSRCC) from
    an audio signal.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        gamma          (float) : power coefficient for resulting energies
                                 Default -1/7.
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.

    Returns:
        (array) : 2d array of MSRCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|^2
    fourrier_transform = rfft(x=windows, n=nfft)
    abs_fft_values = np.abs(fourrier_transform)**2

    # -> x Mel-fbanks
    mel_fbanks_mat = mel_filter_banks(nfilts=nfilts,
                                      nfft=nfft,
                                      fs=fs,
                                      low_freq=low_freq,
                                      high_freq=high_freq,
                                      scale=scale)
    features = np.dot(abs_fft_values, mel_fbanks_mat.T)
    features_no_zero = zero_handling(features)
    
    # -> (.)^(gamma)
    features = features_no_zero**gamma

    # -> DCT(.)
    msrccs = dct(x=features, type=dct_type, axis=1, norm='ortho')[:, :num_ceps]

    # use energy for 1st features column
    if use_energy:
        # compute the power
        power_frames = power_spectrum(fourrier_transform)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        msrccs[:, 0] = np.log(energy)
    msrccs=msrccs.T
    if order_deltas > 0:
        feats = list()
        feats.append(msrccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        feats.append(cmvn(cms(msrccs)))
        msrccs = np.vstack(feats)

    # # liftering
    # if lifter > 0:
    #     msrccs = lifter_ceps(msrccs, lifter)

    # normalization, this post processing will not be used this time.
    # if normalize:
    #     msrccs = cmvn(cms(msrccs))

    
    return msrccs



def imfcc(sig,
          fs=16000,
          num_ceps=20,
          pre_emph=1,
          pre_emph_coeff=0.97,
          win_len=0.02,
          win_hop=0.01,
          win_type="hamming",
          nfilts=32,
          nfft=512,
          low_freq=None,
          high_freq=None,
          scale="constant",
          dct_type=2,
          use_energy=False,
          lifter=22,
          normalize=1,
          order_deltas=2):
    """
    Compute Inverse MFCC features from an audio signal.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.

    Returns:
        (array) : features - the MFFC features: num_frames x num_ceps
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = rfft(x=windows, n=nfft)
    abs_fft_values = np.abs(fourrier_transform)

    #  -> x Mel-fbanks -> log(.) -> DCT(.)
    imel_fbanks_mat = inverse_mel_filter_banks(nfilts=nfilts,
                                               nfft=nfft,
                                               fs=fs,
                                               low_freq=low_freq,
                                               high_freq=high_freq,
                                               scale=scale)
    features = np.dot(abs_fft_values, imel_fbanks_mat.T)

    # -> log(.)
    features_no_zero = zero_handling(features)
    log_features = np.log(features_no_zero)

    # -> DCT(.)
    imfccs = dct(log_features, type=2, axis=1, norm='ortho')[:, :num_ceps]
    imfccs=imfccs.T

    # use energy for 1st features column
    if use_energy:
        # compute the power
        power_frames = power_spectrum(fourrier_transform)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        imfccs[:, 0] = np.log(energy)
    if order_deltas > 0:
        feats = list()
        feats.append(imfccs)
        for d in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        feats.append(cmvn(cms(imfccs)))
        imfccs = np.vstack(feats)
    # # liftering
    # if lifter > 0:
    #     imfccs = lifter_ceps(imfccs, lifter)

    # # normalization
    # if normalize:
    #     imfccs = cmvn(cms(imfccs))
    return imfccs


def mfcc(sig,
         fs=16000,
         num_ceps=20,
         pre_emph=1,
         pre_emph_coeff=0.97,
         win_len=0.02,
         win_hop=0.01,
         win_type="hamming",
         nfilts=32,
         nfft=512,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         use_energy=False,
         lifter=22,
         normalize=1):
    """
    Compute MFCC features (Mel-frequency cepstral coefficients) from an audio
    signal. This function offers multiple approaches to features extraction
    depending on the input parameters. Implemenation is using FFT and based on
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.63.8029&rep=rep1&type=pdf

          - take the absolute value of the FFT
          - warp to a Mel frequency scale
          - take the DCT of the log-Mel-spectrum
          - return the first <num_ceps> components

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.

    Returns:
        (array) : features - the MFFC features: num_frames x num_ceps
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|
    fourrier_transform = rfft(x=windows, n=nfft)
    abs_fft_values = (1 / 1) * np.abs(fourrier_transform)

    #  -> x Mel-fbanks
    mel_fbanks_mat = mel_filter_banks(nfilts=nfilts,
                                      nfft=nfft,
                                      fs=fs,
                                      low_freq=low_freq,
                                      high_freq=high_freq,
                                      scale=scale)
    features = np.dot(abs_fft_values, mel_fbanks_mat.T)

    # -> log(.) -> DCT(.)
    features_no_zero = zero_handling(features)
    log_features = np.log(features_no_zero)
    mfccs = dct(x=log_features, type=dct_type, axis=1,
                norm='ortho')[:, :num_ceps]
    mfccs=mfccs.T
    # use energy for 1st features column
    if use_energy:
        # compute the power
        power_frames = power_spectrum(fourrier_transform)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        mfccs[:, 0] = np.log(energy)
    if 2 > 0:
        feats = list()
        feats.append(mfccs)
        for d in range(2):
            feats.append(Deltas(feats[-1]))
        feats.append(cmvn(cms(mfccs)))
        mfccs = np.vstack(feats)
    # liftering
    # if lifter > 0:
    #     mfccs = lifter_ceps(mfccs, lifter)

    # # normalizatio
    # if normalize:
    #     mfccs = cmvn(cms(mfccs))
    return mfccs


def ngcc(sig,
         fs=16000,
         num_ceps=13,
         pre_emph=1,
         pre_emph_coeff=0.97,
         win_len=0.02,
         win_hop=0.01,
         win_type="hamming",
         nfilts=40,
         nfft=512,
         low_freq=None,
         high_freq=None,
         scale="constant",
         dct_type=2,
         use_energy=False,
         lifter=22,
         normalize=1):
    """
    Compute the normalized gammachirp cepstral coefﬁcients (NGCC features) from an audio signal.

    Args:
        sig            (array) : a mono audio signal (Nx1) from which to compute features.
        fs               (int) : the sampling frequency of the signal we are working with.
                                 Default is 16000.
        num_ceps       (float) : number of cepstra to return.
                                 Default is 13.
        pre_emph         (int) : apply pre-emphasis if 1.
                                 Default is 1.
        pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
                                 Default is 0.97.
        win_len        (float) : window length in sec.
                                 Default is 0.025.
        win_hop        (float) : step between successive windows in sec.
                                 Default is 0.01.
        win_type       (float) : window type to apply for the windowing.
                                 Default is "hamming".
        nfilts           (int) : the number of filters in the filterbank.
                                 Default is 40.
        nfft             (int) : number of FFT points.
                                 Default is 512.
        low_freq         (int) : lowest band edge of mel filters (Hz).
                                 Default is 0.
        high_freq        (int) : highest band edge of mel filters (Hz).
                                 Default is samplerate / 2 = 8000.
        scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
                                 Default is "constant".
        dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
                                 Default is 2.
        use_energy       (int) : overwrite C0 with true log energy
                                 Default is 0.
        lifter           (int) : apply liftering if value > 0.
                                 Default is 22.
        normalize        (int) : apply normalization if 1.
                                 Default is 0.

    Returns:
        (array) : 2d array of NGCC features (num_frames x num_ceps)
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=0.97)

    # -> framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # -> FFT -> |.|**2
    fourrier_transform = rfft(x=windows, n=nfft)
    abs_fft_values = np.abs(fourrier_transform)**2

    #  -> x Gammatone fbanks -> log(.) -> DCT(.)
    gammatone_fbanks_mat = gammatone_filter_banks(nfilts=nfilts,
                                                  nfft=nfft,
                                                  fs=fs,
                                                  low_freq=low_freq,
                                                  high_freq=high_freq,
                                                  scale=scale)

    # compute the filterbank energies
    features = np.dot(abs_fft_values, gammatone_fbanks_mat.T)

    # -> log(.)
    # handle zeros: if feat is zero, we get problems with log
    features_no_zero = zero_handling(x=features)
    log_features = np.log(features_no_zero)

    #  -> DCT(.)
    ngccs = dct(x=log_features, type=dct_type, axis=1,
                norm='ortho')[:, :num_ceps]
    ngccs=ngccs.T
    # use energy for 1st features column
    if use_energy:
        # compute the power
        power_frames = power_spectrum(fourrier_transform)

        # compute total energy in each frame
        frame_energies = np.sum(power_frames, 1)

        # Handling zero enegies
        energy = zero_handling(frame_energies)
        ngccs[:, 0] = np.log(energy)

    if 2 > 0:
        feats = list()
        feats.append(ngccs)
        for d in range(2):
            feats.append(Deltas(feats[-1]))
        feats.append(cmvn(cms(ngccs)))
        ngccs = np.vstack(feats)

    # liftering
    # if lifter > 0:
    #     ngccs = lifter_ceps(ngccs, lifter)

    # # normalization
    # if normalize:
    #     ngccs = cmvn(cms(ngccs))
    return ngccs



# def medium_time_power_calculation(power_stft_signal, M=2):
#     medium_time_power = np.zeros_like(power_stft_signal)
#     power_stft_signal = np.pad(power_stft_signal, [(M, M), (0, 0)], 'constant')
#     for i in range(medium_time_power.shape[0]):
#         medium_time_power[i, :] = sum([
#             1 / float(2 * M + 1) * power_stft_signal[i + k - M, :]
#             for k in range(2 * M + 1)
#         ])
#     return medium_time_power



# def asymmetric_lawpass_filtering(rectified_signal, lm_a=0.999, lm_b=0.5):
#     floor_level = np.zeros_like(rectified_signal)
#     floor_level[0, ] = 0.9 * rectified_signal[0, ]

#     for m in range(floor_level.shape[0]):
#         x = lm_a * floor_level[m - 1, :] + (1 - lm_a) * rectified_signal[m, :]
#         y = lm_b * floor_level[m - 1, :] + (1 - lm_b) * rectified_signal[m, :]
#         floor_level[m, :] = np.where(
#             rectified_signal[m, ] >= floor_level[m - 1, :], x, y)
#     return floor_level



# def temporal_masking(rectified_signal, lam_t=0.85, myu_t=0.2):
#     # rectified_signal[m, l]
#     temporal_masked_signal = np.zeros_like(rectified_signal)
#     online_peak_power = np.zeros_like(rectified_signal)

#     temporal_masked_signal[0, :] = rectified_signal[0, ]
#     online_peak_power[0, :] = rectified_signal[0, :]

#     for m in range(1, rectified_signal.shape[0]):
#         online_peak_power[m, :] = np.maximum(
#             lam_t * online_peak_power[m - 1, :], rectified_signal[m, :])
#         temporal_masked_signal[m, :] = np.where(
#             rectified_signal[m, :] >= lam_t * online_peak_power[m - 1, :],
#             rectified_signal[m, :], myu_t * online_peak_power[m - 1, :])

#     return temporal_masked_signal



# def weight_smoothing(final_output, medium_time_power, N=4, L=128):

#     spectral_weight_smoothing = np.zeros_like(final_output)
#     for m in range(final_output.shape[0]):
#         for l in range(final_output.shape[1]):
#             l_1 = max(l - N, 1)
#             l_2 = min(l + N, L)
#             spectral_weight_smoothing[m, l] = (1 / float(l_2 - l_1 + 1)) * \
#                 sum([(final_output[m, l_] / medium_time_power[m, l_])
#                      for l_ in range(l_1, l_2)])
#     return spectral_weight_smoothing



# def mean_power_normalization(transfer_function,
#                              final_output,
#                              lam_myu=0.999,
#                              L=80,
#                              k=1):
#     myu = np.zeros(shape=(transfer_function.shape[0]))
#     myu[0] = 0.0001
#     normalized_power = np.zeros_like(transfer_function)
#     for m in range(1, transfer_function.shape[0]):
#         myu[m] = lam_myu * myu[m - 1] + \
#             (1 - lam_myu) / L * \
#             sum([transfer_function[m, s] for s in range(0, L - 1)])
#     normalized_power = k * transfer_function / myu[:, None]

#     return normalized_power



# def medium_time_processing(power_stft_signal, nfilts=22):
#     # calculate medium time power
#     medium_time_power = medium_time_power_calculation(power_stft_signal)
#     lower_envelope = asymmetric_lawpass_filtering(medium_time_power, 0.999,
#                                                   0.5)
#     subtracted_lower_envelope = medium_time_power - lower_envelope

#     # half waverectification
#     threshold = 0
#     rectified_signal = np.where(subtracted_lower_envelope < threshold,
#                                 np.zeros_like(subtracted_lower_envelope),
#                                 subtracted_lower_envelope)

#     floor_level = asymmetric_lawpass_filtering(rectified_signal)
#     temporal_masked_signal = temporal_masking(rectified_signal)

#     # switch excitation or non-excitation
#     c = 2
#     F = np.where(medium_time_power >= c * lower_envelope,
#                  temporal_masked_signal, floor_level)

#     # weight smoothing
#     spectral_weight_smoothing = weight_smoothing(F,
#                                                  medium_time_power,
#                                                  L=nfilts)
#     return spectral_weight_smoothing, F



# def pncc(sig,
#          fs=16000,
#          num_ceps=13,
#          pre_emph=0,
#          pre_emph_coeff=0.97,
#          power=2,
#          win_len=0.025,
#          win_hop=0.01,
#          win_type="hamming",
#          nfilts=26,
#          nfft=512,
#          low_freq=None,
#          high_freq=None,
#          scale="constant",
#          dct_type=2,
#          use_energy=False,
#          dither=1,
#          lifter=22,
#          normalize=1):
#     """
#     Compute the power-normalized cepstral coefficients (SPNCC features) from an audio signal.

#     Args:
#         sig            (array) : a mono audio signal (Nx1) from which to compute features.
#         fs               (int) : the sampling frequency of the signal we are working with.
#                                  Default is 16000.
#         num_ceps       (float) : number of cepstra to return.
#                                  Default is 13.
#         pre_emph         (int) : apply pre-emphasis if 1.
#                                  Default is 1.
#         pre_emph_coeff (float) : apply pre-emphasis filter [1 -pre_emph] (0 = none).
#                                  Default is 0.97.
#         power            (int) : spectrum power.
#                                  Default is 2.
#         win_len        (float) : window length in sec.
#                                  Default is 0.025.
#         win_hop        (float) : step between successive windows in sec.
#                                  Default is 0.01.
#         win_type       (float) : window type to apply for the windowing.
#                                  Default is "hamming".
#         nfilts           (int) : the number of filters in the filterbank.
#                                  Default is 40.
#         nfft             (int) : number of FFT points.
#                                  Default is 512.
#         low_freq         (int) : lowest band edge of mel filters (Hz).
#                                  Default is 0.
#         high_freq        (int) : highest band edge of mel filters (Hz).
#                                  Default is samplerate / 2 = 8000.
#         scale           (str)  : choose if max bins amplitudes ascend, descend or are constant (=1).
#                                  Default is "constant".
#         dct_type         (int) : type of DCT used - 1 or 2 (or 3 for HTK or 4 for feac).
#                                  Default is 2.
#         use_energy       (int) : overwrite C0 with true log energy
#                                  Default is 0.
#         dither           (int) : 1 = add offset to spectrum as if dither noise.
#                                  Default is 0.
#         lifter           (int) : apply liftering if value > 0.
#                                  Default is 22.
#         normalize        (int) : apply normalization if 1.
#                                  Default is 0.


#     Returns:
#         (array) : 2d array of PNCC features (num_frames x num_ceps)
#     """
#     # init freqs
#     high_freq = high_freq or fs / 2
#     low_freq = low_freq or 0

#     # run checks
#     if low_freq < 0:
#         raise ParameterError(ErrorMsgs["low_freq"])
#     if high_freq > (fs / 2):
#         raise ParameterError(ErrorMsgs["high_freq"])
#     if nfilts < num_ceps:
#         raise ParameterError(ErrorMsgs["nfilts"])

#     # pre-emphasis
#     if pre_emph:
#         sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

#     # -> STFT()
#     stf_trafo, _ = stft(sig, fs)

#     #  -> |.|^2
#     spectrum_power = np.abs(stf_trafo)**power

#     # -> x Filterbanks
#     gammatone_filter = gammatone_filter_banks(nfilts=nfilts,
#                                               nfft=nfft,
#                                               fs=fs,
#                                               low_freq=low_freq,
#                                               high_freq=high_freq,
#                                               scale=scale)
#     P = np.dot(a=spectrum_power[:, :gammatone_filter.shape[1]],
#                b=gammatone_filter.T)

#     # medium_time_processing
#     S, F = medium_time_processing(P, nfilts=nfilts)

#     # time-freq normalization
#     T = P * S

#     # -> mean power normalization
#     U = mean_power_normalization(T, F, L=nfilts)
#     # -> power law non linearity
#     V = U**(1 / 15)

#     # DCT(.)
#     pnccs = scipy.fftpack.dct(V)[:, :num_ceps]

#     # use energy for 1st features column
#     if use_energy:
#         pspectrum, logE = powspec(sig,
#                                   fs=fs,
#                                   win_len=win_len,
#                                   win_hop=win_hop,
#                                   dither=dither)

#         # bug: pnccs[:, 0] = logE

#     # liftering
#     if lifter > 0:
#         pnccs = lifter_ceps(pnccs, lifter)

#     # normalization
#     if normalize:
#         pnccs = cmvn(cms(pnccs))
#     return pnccs
