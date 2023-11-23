import numpy as np
import random as rd
import matplotlib.pyplot as plt
import time

def main():
    # Initial conditions
    S0 = 10000
    E0 = 1000
    I0 = 100
    R0 = 0
    B=2.75
    G=0.04
    sigma=0.12
    T=120
    dt=0.1
    NT=int(T/dt)
    N=1000

    
    # Time vector  
    t = np.linspace(0,T,NT)

    # Initialize the vectors
    S = np.zeros(NT)
    E = np.zeros(NT)
    I = np.zeros(NT)
    R = np.zeros(NT)

    # Initial conditions    
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0

    

    # Iterate using finite diferences
    for i in range(0,NT-1):

        S[i+1]=S[i]*(-B*I[i]/N*dt + 1)
        E[i+1]=(B*S[i]*I[i]/N - sigma*E[i])*dt + E[i] 
        I[i+1]=(sigma*E[i]-G*I[i])*dt + I[i]
        R[i+1]=G*I[i]*dt + R[i]


if __name__ == "__main__":
    # Begin time count
    start_time = time.time()
    main()
    # End time count
    print("--- %s seconds ---" % (time.time() - start_time))