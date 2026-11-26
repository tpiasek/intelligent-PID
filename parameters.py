K = 2.0        #gain
TA = 3.0       #time constant of tank 1 [s]
TB = 2.0       #time constant of tank 2 [s]
Y0 = 1.0       #nominal operating point

#Sampling
TS = 0.05      #sampling time [s]

#Transport delay
TAU_MAX_S = 1.0                  #maximum delay [s]
TAU_MAX = int(TAU_MAX_S / TS)    #maximum delay in samples (20)

#Control saturation
UMIN = 0.0                       #lower control limit
UMAX_LIST = [1.5, 2.0, 3.0]      #upper limits used in experiments
UMAX_NOSAT = 100.0               #no saturation, for testing

#Setpoints
R_LIST = [0.3, 0.6, 1.0, 1.5, 2.0]

def seconds_to_samples(t_s):
    """Convert time in seconds to a number of samples."""
    return int(round(t_s / TS))