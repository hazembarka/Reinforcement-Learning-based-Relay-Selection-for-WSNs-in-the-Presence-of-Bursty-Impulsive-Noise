"""
Module: DigiCommPy.channels.py
"""
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal
from random import choices
import numpy as np

def awgn(s,SNRdB,L=1):
    """
    AWGN channel
    
    Add AWGN noise to input signal. The function adds AWGN noise vector to signal
    's' to generate a resulting signal vector 'r' of specified SNR in dB. It also
    returns the noise vector 'n' that is added to the signal 's' and the power
    spectral density N0 of noise added
    
    Parameters:
        s : input/transmitted signal vector
        SNRdB : desired signal to noise ratio (expressed in dB)
            for the received signal
        L : oversampling factor (applicable for waveform simulation)
            default L = 1.
    Returns:
        r : received signal vector (r=s+n)
    """
    gamma = 10**(SNRdB/10) #SNR to linear scale
    if s.ndim==1:# if s is single dimensional vector
        P=L*sum(abs(s)**2)/len(s) #Actual power in the vector
    else: # multi-dimensional signals like MFSK
        P=L*sum(sum(abs(s)**2))/len(s) # if s is a matrix [MxN]
        
    N0=P/gamma # Find the noise spectral density    
    if isrealobj(s):# check if input is real/complex object type
        n = sqrt(N0/2)*standard_normal(s.shape) # computed noise
    else:
        n = sqrt(N0/2)*(standard_normal(s.shape)+1j*standard_normal(s.shape))
    r = s + n # received signal    
    return r

def rayleighFading(N,d,path_loss=2):
    """
    Generate Rayleigh flat-fading channel samples
    Parameters:
        N : number of samples to generate
        d : distance between nodes
        path_loss 
    Returns:
        h :Rayleigh flat fading samples
    """
    # 1 tap complex gaussian filter
    sigma = np.sqrt(1/(d**path_loss))
    h = 1/np.sqrt(2)*np.random.normal(0,sigma,N)+ 1j*np.random.normal(0,sigma,N)* 1/np.sqrt(2)
    return h.round(3)



def ricianFading(K_dB,N):
    """
    Generate Rician flat-fading channel samples
    Parameters:
        K_dB: Rician K factor in dB scale
        N : number of samples to generate
    Returns:
        abs_h : Rician flat fading samples
    """
    K = 10**(K_dB/10) # K factor in linear scale
    mu = sqrt(K/(2*(K+1))) # mean
    sigma = sqrt(1/(2*(K+1))) # sigma
    h = (sigma*standard_normal(N)+mu)+1j*(sigma*standard_normal(N)+mu)
    return abs(h)




def TSMG(s,P_B,NOISE_MEMORY,R,SNRdB):

    SNR_lin = 10**(SNRdB/10) 
    nb_of_symbols = len(s)

    P=sum(abs(s)**2)/len(s) #Actual power in the vector
        
    SIGMA2_G=P/SNR_lin # Find the noise spectral density    
    
    # list to contain noise signal :
    n = [] 
    states = []

    #Initial state :
    TSMG_states = ['B','G']
    initial_probs = [0.5 , 0.5]
    previous_state = choices(TSMG_states,initial_probs)[0]

    #computing the transition probabilities :
    P_GB = P_B / NOISE_MEMORY
    P_BG = 1 / NOISE_MEMORY - P_GB

    # Repeting Process for each symbol :
    for i in range(nb_of_symbols): 

      #Getting the current transition probabilities :
      if previous_state == 'G':
        transition_probs = [P_GB , 1-P_GB]
        
      elif previous_state == 'B':

        transition_probs = [1-P_BG , P_BG]

      ## getting the state based on the current state and the transition probabilities
      state = choices(TSMG_states,transition_probs)[0]

      N0 = SIGMA2_G
      if state == 'B':
        N0 = SIGMA2_G*R
        previous_state = 'B'
        states.append(0)

      else:
        previous_state = 'G'
        states.append(1)

      n.append(np.sqrt(N0/2)*(standard_normal()+1j*standard_normal()))

    n = np.array(n).round(3)
    r = s + n # received signal    

    return r, np.array(states)