import numpy as np
import pandas as pd
import glob

input_dir = "/home/mscherbela/develop/LaplaceSolverMPS/outputs/run5_eps1e-12_swp150_rank200"
input_files = glob.glob(input_dir + "/*/TM.out")
data_list = []
for fname in input_files:
    L = int(fname.split('/')[-2])
    is_bpx = True
    with open(fname) as f:
        for l in f:
            if l.startswith("L = "):
                data = dict(L=L)
            if l.startswith("L2 delta_u:"):
                data['L2_delta_u'] = float(l.split("L2 delta_u:")[-1])
            if l.startswith("H1 delta_u:"):
                data['H1_delta_u'] = float(l.split("H1 delta_u:")[-1])
            if l.startswith("Error of H1 norm:"):
                data['error_H1_norm'] = float(l.split("Error of H1 norm:")[-1])
            if l.startswith("Error of L2 norm:"):
                data['error_L2_norm'] = float(l.split("Error of L2 norm:")[-1])
                data['bpx'] = is_bpx
                data_list.append(data)
                is_bpx = not is_bpx
df = pd.DataFrame(data_list)
output_fname =f"/home/mscherbela/develop/LaplaceSolverMPS/outputs/{input_dir.split('/')[-1]}.csv"
df.sort_values(['bpx', 'L'], inplace=True)
df.to_csv(output_fname, index=False)







