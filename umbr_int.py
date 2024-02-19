import numpy as np
import math
import toml
import sys

class Window:
    def __init__(self, x0, k, filename, options):
        self.x0 = x0
        self.k = k
        self.filename = filename
        self.mean = 0
        self.std = 0
        self.num = 0
        self.read_data(options)

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
        data = np.array(data)  # Convert list to NumPy array
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.num = len(data)
        print(f'{self.num} values read after filtering')

class Options:
    def __init__(self, config):
        self.Nbin = config['Nbin']
        self.bin_min = config['bin_min']
        self.bin_max = config['bin_max']
        self.Nwin = len(config['windows'])
        self.Temperature = config['Temperature']
        self.first_time = config['first_time']
        self.out_file = config['out_file']
        self.kt_units = config['kt_units']
        self.zero_point = config['zero_point']

def read_parameters(config):
    options = Options(config)
    windows = []
    for window_config in config['windows']:
        windows.append(Window(window_config['pos'], window_config['k'], window_config['file'], options))
    return options, windows

def Pb(w, x, mean, std, sqrt_two_pi):
    return 1 / (std * sqrt_two_pi) * np.exp(-0.5 * ((x - mean) / std)**2)

def umb_int(param_file_path):
    with open(param_file_path, 'r') as file:
        config = toml.load(file)
        
    options,windows = read_parameters(config)
    
    sqrt_two_pi = math.sqrt(math.tau)
    kb = 0.0083144621  # Boltzmann constant in kJ/(mol*K)
    kbT = kb * options.Temperature
    bin_sz = (options.bin_max - options.bin_min) / options.Nbin

    dAu = np.zeros((options.Nwin, options.Nbin))
    dAfinal = np.zeros(options.Nbin)
    Afinal = np.zeros(options.Nbin)

    for i, window in enumerate(windows):
        for j in range(options.Nbin):
            x = options.bin_min + bin_sz * (j + 0.5)
            dAu[i, j] = kbT * ((x - window.mean) / (window.std**2)) - window.k * (x - window.x0)

    for j in range(options.Nbin):
        x = options.bin_min + bin_sz * (j + 0.5)
        weights = np.array([window.num * Pb(i, x, window.mean, window.std, sqrt_two_pi) for i, window in enumerate(windows)])
        dAfinal[j] = np.sum(weights * dAu[:, j]) / np.sum(weights)

    Afinal[0] = 0
    for j in range(1, options.Nbin):
        Afinal[j] = Afinal[j-1] + bin_sz * 0.5 * (dAfinal[j-1] + dAfinal[j])

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

    # Writing output with an explicit loop
    with open(options.out_file, 'w') as f:
        for i in range(options.Nbin):
            bin_center = options.bin_min + bin_sz * (i + 0.5)
            f.write(f'{bin_center:10.5f} {Afinal[i]:10.5f}\n')

# Replace 'gpt.toml' with the path to your TOML parameter file
umb_int(sys.argv[1])

