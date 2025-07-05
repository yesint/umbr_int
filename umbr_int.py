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


    def interval_mean_std(self, i: int):
        span = self.intervals[i] if i!=None else self.data[:]    
        self.mean = np.mean(span)
        self.std = np.std(span)
        self.num = len(span)
    

class Config:
    def __init__(self, toml_file: str):
        toml = tomllib.load(open(toml_file, 'rb'))
        self.first_time = toml.get('first_time',-1.0)
        
        self.windows = []
        for w in toml['windows']:
            self.windows.append(Window(w['pos'], w['k'], w['file'], self.first_time))

        # Convert angular force constant if asked
        if toml.get('k_rad2_to_deg2',False):
            print('Convertin angular force constants: kJ mol^-1 rad^-2 --> kJ mol^-1 deg^-2')
            for w in self.windows:
                # Convert kJ mol^-1 rad^-2 --> kJ mol^-1 deg^-2
                w.k *= 0.0003046174

        self.Nintervals = toml.get('Nintervals',1)
        intervals_type = toml.get('intervals_type','window_time')
        
        if self.Nintervals>1:
            if intervals_type == 'common_time':
                print('Defining intervals by common time of all windows')
                # Find common time interval for all windows
                min_t = np.min([w.time[0] for w in self.windows])
                max_t = np.max([w.time[-1] for w in self.windows])

                if max_t - min_t <= 0:
                    raise Exception(f"Windows don't cover a common time interval!")

                print(f'Windows cover common time interval {min_t}..{max_t}')
                t_intervals = np.linspace(min_t,max_t,self.Nintervals)
                
                # Set index intervals for windows
                for w in self.windows:
                    split_points = []
                    for ti in t_intervals:
                        while w.time[cur] < ti[0]:
                            cur += 1
                        split_points.append(cur)
                    # This returns array views so no data is copied
                    w.intervals = np.array_split(w.data, split_points)[1:-1]
            else:
                print('Defining intervals independently for each window')
                for w in self.windows:
                    # This returns array views so no data is copied
                    w.intervals = np.array_split(w.data, self.Nintervals)
            
        # Compute data min and max        
        w_pos = [w.pos for w in self.windows]
        self.bin_min = np.min(w_pos)
        self.bin_max = np.max(w_pos)

        self.Nwin = len(self.windows)
        self.Temperature = toml.get('Temperature',300.0)
        
        self.out_file = toml.get('out_file','pmf.dat')
        self.kt_units = toml.get('kt_units',False)
        self.zero_point = toml.get('zero_point','left')
        
        self.Nbin = toml.get('Nbin',100)
        self.bin_sz = (self.bin_max - self.bin_min) / self.Nbin


def Pb(x: float, mean: float, std: float) -> float:
    return 1.0 / (std * sqrt_two_pi) * np.exp(-0.5 * ((x - mean) / std)**2)


def compute_interval(config: Config, interval: int|None) -> np.ndarray:
    if interval is not None:
        print(f'Computing PMF for interval {interval}...')
    else:
        print(f'Computing whole PMF...')

    kbT = kb * config.Temperature    

    dAu = np.zeros((config.Nwin, config.Nbin))
    dAfinal = np.zeros(config.Nbin)
    Afinal = np.zeros(config.Nbin)

    for w_ind, window in enumerate(config.windows):
        # Get averages for given interval
        window.interval_mean_std(interval)
        
        # Compute dAu
        for j in range(config.Nbin):
            x = config.bin_min + config.bin_sz * (j + 0.5)
            dAu[w_ind, j] = kbT * ((x - window.mean) / (window.std**2)) - window.k * (x - window.pos)

    for j in range(config.Nbin):
        x = config.bin_min + config.bin_sz * (j + 0.5)
        weights = np.array([w.num * Pb(x, w.mean, w.std) for w in config.windows])
        dAfinal[j] = np.sum(weights * dAu[:, j]) / np.sum(weights)

    Afinal[0] = 0.0
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


def write_pmf(outfile: str, pmf: np.ndarray, config: Config, errors: np.ndarray|None):
    with open(outfile, 'w') as f:
        for i in range(config.Nbin):
            bin_center = config.bin_min + config.bin_sz * (i + 0.5)
            if errors is None:
                f.write(f'{bin_center:10.5f} {pmf[i]:10.5f}\n')
            else:
                f.write(f'{bin_center:10.5f} {pmf[i]:10.5f} {errors[i]:10.5f}\n')


####################
# Run
####################
config = Config(sys.argv[1])

# Compute whole PMF
whole_pmf = compute_interval(config,None)

pmfs = np.zeros((config.Nbin, config.Nintervals))
rmsd_pmfs = np.zeros((config.Nbin, config.Nintervals))

for i in range(config.Nintervals):
    pmfs[:,i] = compute_interval(config,i)
    
    # Align to the whole pmf
    optimal_c = np.mean(whole_pmf - pmfs[:,i])
    rmsd_pmfs[:,i] = pmfs[:,i] + optimal_c
    
    # Writing individual intervals
    stem = Path(config.out_file).stem
    ext = Path(config.out_file).suffix
    write_pmf(f'{stem}_{i}{ext}', rmsd_pmfs[:,i], config, errors=None)

# Write whole PMF
if config.Nintervals>1:
    # Compute errors in each point
    errs = np.zeros(config.Nbin)
    for i in range(config.Nbin):    
        errs[i] = np.std(rmsd_pmfs[i,:])

    write_pmf(config.out_file, whole_pmf, config, errs)
else:
    write_pmf(config.out_file, whole_pmf, config, errors=None)
