import numpy as np
import math
import toml
import sys
from pathlib import Path

sqrt_two_pi = math.sqrt(math.tau)

class Window:
    def __init__(self, x0, k, filename, options):
        self.x0 = x0
        self.k = k
        self.filename = filename
        self.mean = 0
        self.std = 0
        self.num = 0
        self.read_data(options)

    def get_interval(self,interval,Nintervals):
        if Nintervals>0 and interval>=0:
            interval_sz = len(self.data)/Nintervals
            b = int(interval_sz*interval)
            e = int(interval_sz*(interval+1))
            self.mean = np.mean(self.data[b:e])
            self.std = np.std(self.data[b:e])
            self.num = len(self.data[b:e])
        else:
            self.mean = np.mean(self.data)
            self.std = np.std(self.data)
            self.num = len(self.data)

    def read_data(self, options):
        print(f'Reading data file {self.filename} starting time: {options.first_time}...')
        data = []  # List to store filtered data
        with open(self.filename, 'r') as file:
            for line in file:
                if line.startswith(('#', '@')):
                    continue  # Skip comments and header lines
                time, x = map(float, line.strip().split()[:2])
                if time >= options.first_time:
                    data.append(x)  # Add x value if time >= first_time
        self.data = np.array(data)  # Convert list to NumPy array and store
        print(f'{len(data)} values read after filtering')


class Options:
    def __init__(self, config):
        self.Nbin = config['Nbin']

        # Compute min and max        
        minval = 1.0e6
        maxval = -1.0e6
        for w in config['windows']:
            if w["pos"] > maxval:
                maxval = w["pos"]
            if w["pos"] < minval:
                minval = w["pos"]
                
        self.bin_min = minval
        self.bin_max = maxval
        
        self.Nwin = len(config['windows'])
        self.Temperature = config['Temperature']
        self.first_time = config['first_time']
        self.Nintervals = config['Nintervals']
        self.out_file = config['out_file']
        self.kt_units = config['kt_units']
        self.zero_point = config['zero_point']
        self.cylindrical = config['cylindrical']
        
        self.bin_sz = (self.bin_max - self.bin_min) / self.Nbin


def read_parameters(config):
    options = Options(config)
    windows = []
    for window_config in config['windows']:
        windows.append(Window(window_config['pos'], window_config['k'], window_config['file'], options))
    return options, windows


def Pb(x, mean, std):
    return 1 / (std * sqrt_two_pi) * np.exp(-0.5 * ((x - mean) / std)**2)


def umb_int_init(param_file_path):
    with open(param_file_path, 'r') as f:
        config = toml.load(f)
        
    options,windows = read_parameters(config)
    return options,windows
    

def compute_interval(options,windows,interval):
    if interval>=0:
        print(f'Computing PMF for interval {interval}...')
    else:
        print(f'Computing PMF for whole duration...')
    
    
    kb = 0.0083144621  # Boltzmann constant in kJ/(mol*K)
    kbT = kb * options.Temperature    

    dAu = np.zeros((options.Nwin, options.Nbin))
    dAfinal = np.zeros(options.Nbin)
    Afinal = np.zeros(options.Nbin)

    for i, window in enumerate(windows):
        # Get averages for given interval
        window.get_interval(interval,options.Nintervals)
        # Compute dAu
        for j in range(options.Nbin):
            x = options.bin_min + options.bin_sz * (j + 0.5)
            dAu[i, j] = kbT * ((x - window.mean) / (window.std**2)) - window.k * (x - window.x0)

    for j in range(options.Nbin):
        x = options.bin_min + options.bin_sz * (j + 0.5)
        if options.cylindrical:
            weights = np.zeros((len(windows),))
            for i, window in enumerate(windows):
                factor = 1.0/window.x0 if window.x0>0 else 1.0                
                weights[i] = window.num * factor * Pb(x, window.mean, window.std)                
        else:
            weights = np.array([window.num * Pb(x, window.mean, window.std) for i, window in enumerate(windows)])
        dAfinal[j] = np.sum(weights * dAu[:, j]) / np.sum(weights)

    Afinal[0] = 0
    for j in range(1, options.Nbin):
        Afinal[j] = Afinal[j-1] + options.bin_sz * 0.5 * (dAfinal[j-1] + dAfinal[j])

    # Set zero as asked
    if options.zero_point == 'min':
        Afinal -= np.min(Afinal)
    elif options.zero_point == 'left':
        Afinal -= Afinal[0]
    elif options.zero_point == 'right':
        Afinal -= Afinal[-1]
    
    # Convert to kT if asked
    if options.kt_units:
        Afinal /= kbT
   
    return Afinal


def write_pmf(outfile,pmf,options,errors):
    with open(outfile, 'w') as f:
        for i in range(options.Nbin):
            bin_center = options.bin_min + options.bin_sz * (i + 0.5)
            if len(errors) == 0:
                f.write(f'{bin_center:10.5f} {pmf[i]:10.5f}\n')
            else:
                f.write(f'{bin_center:10.5f} {pmf[i]:10.5f} {errors[i]:10.5f}\n')


####################
# Run
####################

options,windows = umb_int_init(sys.argv[1])

if options.Nintervals>1:
    pmfs = np.zeros((options.Nbin, options.Nintervals))
    for interval in range(options.Nintervals):
        pmfs[:,interval] = compute_interval(options,windows,interval)
        # Writing output
        fname = Path(options.out_file).stem
        ext = Path(options.out_file).suffix
        outfile = f'{fname}_{interval}{ext}'
        write_pmf(outfile,pmfs[:,interval],options,errors=[])
    
    # Compute errors in each point
    errs = np.zeros(options.Nbin)
    for i in range(options.Nbin):    
        errs[i] = np.std(pmfs[i,:])
    
# Compute whole    
whole_pmf = compute_interval(options,windows,-1)
# Writing output
write_pmf(options.out_file,whole_pmf,options,errs)

