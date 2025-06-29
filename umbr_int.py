import numpy as np
import math
import tomllib
import sys
from pathlib import Path

sqrt_two_pi = math.sqrt(math.tau)
kb = 0.0083144621  # Boltzmann constant in kJ/(mol*K)

"Single umbrella window"
class Window:
    def __init__(self, pos, k, filename, first_time):
        self.pos = pos
        self.k = k
        
        self.mean = 0
        self.std = 0
        self.num = 0

        print(f'Reading file "{filename}"...')
        time = []
        data = []
        for i,line in enumerate(open(filename).readlines()):
            line = line.strip()
            if line.startswith(('#','@')):
                continue
            
            fields = line.split()
            if len(fields)<2:
                print(f'\tLine {i} is corrupted, skipping')
                continue
            
            t = float(fields[0])
            if t>=first_time:
                time.append(t)
                data.append(float(fields[1]))

        if len(data) == 0:
            raise Exception('No values read. first_time is too large or file is empty.')
        
        self.time = np.array(time, dtype=float)
        self.data = np.array(data, dtype=float)
    
        print(f'\t{len(self.data)} values, t={self.time[0]}..{self.time[-1]}')


    def interval_mean_std(self, i):
        if i != None:
            t_b, t_e = self.intervals[i]
            mask = np.logical_and(self.time >= t_b, self.time < t_e)
            self.mean = np.mean(self.data[mask])
            self.std = np.std(self.data[mask])
            self.num = np.count_nonzero(mask)
        else:
            self.mean = np.mean(self.data)
            self.std = np.std(self.data)
            self.num = len(self.data)
    

class Config:
    def __init__(self, toml):
        self.first_time = toml.get('first_time',-1.0)
        
        self.windows = []
        for w in toml['windows']:
            self.windows.append(Window(w['pos'], w['k'], w['file'], self.first_time))

        self.Nintervals = toml.get('Nintervals',1)
        if self.Nintervals>1:
            #if toml['intervals_type'] == 'common_time':
                print('Defining intervals by common time of all windows')
                intervals = []
                # Find common time interval for all windows
                min_times = []
                max_times = []
                
                for w in self.windows:
                    min_times.append(w.time[0])
                    max_times.append(w.time[-1])
                
                min_t = np.max(np.array(min_times))
                max_t = np.min(np.array(max_times))

                if max_t - min_t <= 0:
                    raise Exception(f"Windows don't cover a common time interval!")

                print(f'Windows cover common time interval {min_t}..{max_t}')
                dt = (max_t-min_t)/self.Nintervals
                print(f'Ther are {self.Nintervals} intervals, of length {dt}')
                
                for interval in range(self.Nintervals):
                    b = min_t + dt*interval
                    e = min_t + dt*(interval+1)
                    intervals.append([b,e])
                
                # Set index intervals for windows
                for w in self.windows:
                    w.intervals = intervals
            # else:
            #     print('Defining intervals independently for each window')
            #     for w in self.windows:
            #         w.intervals = intervals

            
        # Compute data min and max        
        self.bin_min = sys.float_info.max
        self.bin_max = sys.float_info.min
        for w in self.windows:
            if w.pos > self.bin_max:
                self.bin_max = w.pos
            if w.pos < self.bin_min:
                self.bin_min = w.pos
        
        self.Nwin = len(self.windows)
        self.Temperature = toml.get('Temperature',300.0)
        
        self.out_file = toml.get('out_file','pmf.dat')
        self.kt_units = toml.get('kt_units',False)
        self.zero_point = toml.get('zero_point','left')
        
        self.Nbin = toml.get('Nbin',100)
        self.bin_sz = (self.bin_max - self.bin_min) / self.Nbin


def Pb(x, mean, std):
    return 1 / (std * sqrt_two_pi) * np.exp(-0.5 * ((x - mean) / std)**2)


def compute_interval(config: Config, i):
    if i != None:
        print(f'Computing PMF for interval {i}...')
    else:
        print(f'Computing whole PMF...')

    kbT = kb * config.Temperature    

    dAu = np.zeros((config.Nwin, config.Nbin))
    dAfinal = np.zeros(config.Nbin)
    Afinal = np.zeros(config.Nbin)

    for w, window in enumerate(config.windows):
        # Get averages for given interval
        window.interval_mean_std(i)
        
        # Compute dAu
        for j in range(config.Nbin):
            x = config.bin_min + config.bin_sz * (j + 0.5)
            dAu[w, j] = kbT * ((x - window.mean) / (window.std**2)) - window.k * (x - window.pos)

    for j in range(config.Nbin):
        x = config.bin_min + config.bin_sz * (j + 0.5)
        weights = np.array([window.num * Pb(x, window.mean, window.std) for window in config.windows])
        dAfinal[j] = np.sum(weights * dAu[:, j]) / np.sum(weights)

    Afinal[0] = 0
    for j in range(1, config.Nbin):
        Afinal[j] = Afinal[j-1] + config.bin_sz * 0.5 * (dAfinal[j-1] + dAfinal[j])

    # Set zero as asked
    if config.zero_point == 'min':
        Afinal -= np.min(Afinal)
    elif config.zero_point == 'left':
        Afinal -= Afinal[0]
    elif config.zero_point == 'right':
        Afinal -= Afinal[-1]
    
    # Convert to kT if asked
    if config.kt_units:
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

toml = tomllib.load(open(sys.argv[1], 'rb'))
config = Config(toml)

# Compute whole PMF
whole_pmf = compute_interval(config,None)

pmfs = np.zeros((config.Nbin, config.Nintervals))
rmsd_pmfs = np.zeros((config.Nbin, config.Nintervals))

for i in range(config.Nintervals):
    pmfs[:,i] = compute_interval(config,i)
    
    # RMSD align to the whole pmf
    optimal_c = np.mean(whole_pmf - pmfs[:,i])
    rmsd_pmfs[:,i] = pmfs[:,i] + optimal_c
    
    # Writing individual intervals
    stem = Path(config.out_file).stem
    ext = Path(config.out_file).suffix
    outfile = f'{stem}_{i}{ext}'
    write_pmf(outfile, rmsd_pmfs[:,i], config, errors=[])

# Write whole PMF
if config.Nintervals>1:
    # Compute errors in each point
    errs = np.zeros(config.Nbin)
    for i in range(config.Nbin):    
        errs[i] = np.std(rmsd_pmfs[i,:])

    write_pmf(config.out_file, whole_pmf, config, errs)
else:
    write_pmf(config.out_file, whole_pmf, config, errors=[])
