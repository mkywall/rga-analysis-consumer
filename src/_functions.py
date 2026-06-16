import os
import shutil
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
import numpy as np
from datetime import datetime
import zipfile
from collections import defaultdict

def make_subfolders(output_folder_ascii):
    #output_folder_ascii = os.path.join(folder_path, "Analysis_results-ascii")
    subfolders = [
        "TEY_normalized",
        "TEY_normalized_averaged",
        "MS",
        "MS_averaged",
        "MS(t)",
        "MS(t)_averaged",
        "Total_outgassing",
        "Total_outgassing_averaged"
    ]

    # Create output folder and subfolders if they don't exist
    os.makedirs(output_folder_ascii, exist_ok=True)
    for sf in subfolders:
        os.makedirs(os.path.join(output_folder_ascii, sf), exist_ok=True)

    print('subfolder generated')
    return 0


def clean_folder(base_folder):
    # Folders to delete
    for folder_name in ["plots", "rawdataplots", "outgassing_data"]:
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            print(f"Deleted folder: {folder_path}")
        else:
            print(f"No folder '{folder_name}' found in {folder_path}")



def convert_time_to_seconds(time_str_array):
    """
    Convert an array of time strings (format '%Y/%m/%d %H:%M:%S.%f') to seconds relative to the first timestamp.
    """
    start_date = datetime.strptime(time_str_array[0], '%Y/%m/%d %H:%M:%S.%f')
    return np.array([
        (datetime.strptime(dt, '%Y/%m/%d %H:%M:%S.%f') - start_date).total_seconds()
        for dt in time_str_array
    ])
# Define absolute times (in seconds)
BEAM_ON_USED_S          = 30.0  # how many seconds of beam-on intervals to include for determining values for RGA spectrum
BEAM_OFF_BEFORE_S       = 20.0  # how many seconds (before beam-on) to exclude for the first beam-off region
BEAM_OFF_AFTER_S        = 30.0  # how many seconds (after beam-on) to exclude for the second beam-off region

def get_beam_intervals_in_seconds(time_array, beam_on_indices,BEAM_ON_USED_S=BEAM_ON_USED_S,BEAM_OFF_BEFORE_S=BEAM_OFF_BEFORE_S,BEAM_OFF_AFTER_S=BEAM_OFF_AFTER_S):
    """
    Given an array of times (seconds) and the indices where the shutter is on,
    return [start_idx, end_idx] for the beam-on region
    and two beam-off regions in seconds.
    """
    # Indices of first/last beam-on
    first_on_idx = beam_on_indices[0]
    last_on_idx  = beam_on_indices[-1]

    # Times (in seconds) of the first/last beam on
    time_first_on = time_array[first_on_idx]
    time_last_on  = time_array[last_on_idx]

    # --- Beam ON interval ---
    beam_on_start_time = time_first_on  # no extra margin at the start
    beam_on_end_time   = time_first_on + BEAM_ON_USED_S  # use only first X seconds
    on_start_idx = np.searchsorted(time_array, beam_on_start_time, side='left')
    on_end_idx   = np.searchsorted(time_array, beam_on_end_time,   side='right')
    beam_on_interval = [on_start_idx, on_end_idx]

    # --- Beam OFF 1 interval ---
    # from t=0 up to (time_first_on - BEAM_OFF_BEFORE_S)
    beam_off1_end_time = time_first_on - BEAM_OFF_BEFORE_S
    off1_end_idx = np.searchsorted(time_array, beam_off1_end_time, side='right')
    beam_off1_interval = [1, off1_end_idx]

    # --- Beam OFF 2 interval ---
    # from (time_last_on + BEAM_OFF_AFTER_S) to the end
    beam_off2_start_time = time_last_on + BEAM_OFF_AFTER_S
    off2_start_idx = np.searchsorted(time_array, beam_off2_start_time, side='left')
    beam_off2_interval = [off2_start_idx, len(time_array) - 1]

    return beam_on_interval, beam_off1_interval, beam_off2_interval

def extract_sample_name(filename):
    """
    Extract the sample name from the filename.
    Assumes the pattern: <SampleName>_RGA_ or <SampleName>_TEY_
    """
    for tag in ["_RGA_", "_TEY_"]:
        if tag in filename:
            return filename.split(tag)[0].strip()
    return None

def parse_photodiode_current(TEY_filename):
    """
    Extract the photodiode current (in microamps) from the TEY filename.
    Looks for a substring between '_PD_' and 'uA'. If not found, returns None.
    Example: "Sample1_TEY_PD_0.3uA_" -> returns 0.3
    """
    if "_PD_" in TEY_filename and "uA" in TEY_filename:
        part = TEY_filename.split("_PD_")[1]
        pd_str = part.split("uA")[0]
        try:
            return float(pd_str)
        except ValueError:
            return None
    return None

def determine_intervals(TEY_file, rga_file):
    """
    Determine the beam-on indices by comparing the TEY (pressure-current) data and RGA timestamps.
    """
    TEY_data = np.loadtxt(TEY_file, skiprows=1, delimiter='\t', dtype=float)
    rga_data = np.loadtxt(rga_file, skiprows=2, delimiter='\t', dtype=str)
    
    TEY_time = TEY_data[:, 0]
    shutter  = TEY_data[:, 2]
    
    rga_time_str = rga_data[:, 0]
    rga_time = np.array([datetime.strptime(t, '%Y/%m/%d %H:%M:%S.%f') for t in rga_time_str])
    
    beam_on_indices = []
    for idx, rga_time_val in enumerate(rga_time):
        # Find the index of the nearest TEY time value
        nearest_idx = np.abs(TEY_time - (rga_time_val - rga_time[0]).total_seconds()).argmin()
        if shutter[nearest_idx] == 1:
            beam_on_indices.append(idx)
                         
    return beam_on_indices

def process_and_plot_column(data, column_to_plot, sample_name, beam_on_indices):
    """
    Process a given RGA mass channel (column), perform background subtraction based on beam-off intervals,
    and plot the corrected time trace. Returns:
       - beam_on_interval, beam_off1_interval, beam_off2_interval,
       - [avg_value]  (avg over beam-on region)
       - [std_value]  (std from beam-off region)
       - time_array: time (in seconds) for the measurement
       - corrected_data: background-corrected pressure time series
    """
    #fig, ax = plt.subplots(figsize=(8, 6))
    
    first_column = data[:, 0]
    column_data = data[:, column_to_plot].astype(float)
    time_array = convert_time_to_seconds(first_column)
    
    # Compute beam intervals
    beam_on_interval, beam_off1_interval, beam_off2_interval = get_beam_intervals_in_seconds(time_array, beam_on_indices)
    
    # Fit linear background using beam-off data
    x_fit = np.concatenate((
        time_array[beam_off1_interval[0]:beam_off1_interval[1]],
        time_array[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    y_fit = np.concatenate((
        column_data[beam_off1_interval[0]:beam_off1_interval[1]],
        column_data[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    regression_coefficients = np.polyfit(x_fit, y_fit, 1)
    background_line = np.polyval(regression_coefficients, time_array)
    
    corrected_data = column_data - background_line
    #ax.plot(time_array, corrected_data, label=f"m/z = {column_to_plot}")
    
    # Compute average (over beam-on region) and std (from beam-off region)
    data_beam_on = corrected_data[beam_on_interval[0]: beam_on_interval[1]]
    data_beam_off = np.concatenate((
        corrected_data[beam_off1_interval[0]:beam_off1_interval[1]],
        corrected_data[beam_off2_interval[0]:beam_off2_interval[1]]
    ))
    avg_value = np.mean(data_beam_on)
    std_value = np.std(data_beam_off)
        
    return beam_on_interval, beam_off1_interval, beam_off2_interval, [avg_value], [std_value], time_array, corrected_data

def compress_folder(base_folder):
    # Subfolders to compress
    subfolders = ["outgassing_data", "plots", "rawdataplots"]
    zip_path = os.path.join(base_folder, "outgassing_data_zip.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for sub in subfolders:
            folder_path = os.path.join(base_folder, sub)
            if os.path.isdir(folder_path):
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, base_folder)
                        zipf.write(file_path, arcname)
                print(f"Added '{sub}' to {zip_path}")
            else:
                print(f"No folder '{sub}' found in {base_folder}")

    print(f"All available subfolders compressed into {zip_path}")

def round_sig(x, sig=5):
    return np.round(x, sig - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0

def fix_spikes_with_time(time_array, data_array, start_time=62, threshold=3):
    """
    Fix spikes only for data points where time > start_time.
    """
    data_fixed = data_array.copy()
    # Get indices where time > start_time
    indices = np.where(time_array > start_time)[0]
    # Only check spikes from second to second-last in this subset to have neighbors
    for i in indices:
        if i == 0 or i == len(data_array) - 1:
            continue  # skip first and last index (no two neighbors)
        if time_array[i] <= start_time:
            continue
        
        prev_val = data_fixed[i - 1]
        curr_val = data_fixed[i]
        next_val = data_fixed[i + 1]
        
        neighbors_avg = (prev_val + next_val) / 2
        
        if neighbors_avg != 0:
            diff_ratio = abs(curr_val - neighbors_avg) / abs(neighbors_avg)
        else:
            diff_ratio = abs(curr_val - neighbors_avg)
        
        diff_prev = abs(curr_val - prev_val)
        diff_next = abs(curr_val - next_val)
        
        if diff_ratio > threshold and diff_prev > threshold * abs(prev_val) and diff_next > threshold * abs(next_val):
            data_fixed[i] = neighbors_avg
            
    return data_fixed

def save_mass_spectra_with_pandas(data_dict, output_dir):
    """
    Save averaged mass spectra to text files using pandas.
    
    Parameters:
    - data_dict: dict of {sample_name: {'avg': [...], 'std': [...]}}
    - output_dir: folder where files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    for sample_name, values in data_dict.items():
        avg = values.get('avg', [])
        std = values.get('std', [])
        if len(avg) != len(std):
            raise ValueError(f"Length mismatch in sample '{sample_name}': avg({len(avg)}) vs std({len(std)})")

        df = pd.DataFrame({
            "MZ": range(1, len(avg) + 1),
            "Pressure(Torr)": avg,
            "Std(Torr)": std
        })

        filename = f"{sample_name}_MS.txt"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath, sep='\t', index=False, float_format="%.6e")
        print(f"Saved: {filepath}")


def save_sample_ion_to_txt(sample_ion, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for sample_name, mz_dict in sample_ion.items():
        # Create a filtered copy without 'sum_std'
        filtered_mz_dict = {k: v for k, v in mz_dict.items() if k != 'sum_std'}

        first_mz = next(iter(filtered_mz_dict))
        time_array = filtered_mz_dict[first_mz][0]

        # Start with time column
        columns = [pd.Series(time_array, name="Time(s)")]

        # Add mz and std columns in sorted order
        for mz in sorted(filtered_mz_dict.keys()):
            time, corrected, std = filtered_mz_dict[mz]

            if not (len(time) == len(corrected) == len(std)):
                raise ValueError(f"Length mismatch in sample '{sample_name}', m/z {mz}")

            columns.append(pd.Series(corrected, name=f"MZ{mz}(Torr)"))
            columns.append(pd.Series(std, name=f"Std{mz}(Torr)"))

        # Concatenate all columns at once
        df = pd.concat(columns, axis=1)

        # Save to file
        filename = f"{sample_name}_MS_t.txt"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, sep='\t', index=False, float_format="%.6e")
        print(f"Saved: {filepath}")


def save_sample_ion_to_total_outgassing_txt(sample_ion, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for sample_name, mz_dict in sample_ion.items():
        first_mz = next(iter(mz_dict))
        time_array = mz_dict[first_mz][0]

        mz_values = []
        std_values = []

        for mz in range(1, 201):  # Loop over mz1 to mz200
            if mz in mz_dict:
                time, corrected, std = mz_dict[mz]

                if not (len(time) == len(corrected) == len(std)):
                    raise ValueError(f"Length mismatch in sample '{sample_name}', m/z {mz}")

                mz_values.append(corrected)
                std_values.append(std)

        # Stack and compute mean across m/z channels
        mz_array = np.array(mz_values)  # shape: (num_mz_present, num_time)
        std_array = np.array(std_values)

        mean_mz = np.sum(mz_array, axis=0)
        mean_std = sample_ion[sample_name]['sum_std']  #np.mean(std_array, axis=0)

        # Build DataFrame
        df = pd.DataFrame({
            "Time(s)": [f"{t:.6f}".rstrip('0').rstrip('.') for t in time_array],
            "Pressure(Torr)": mean_mz,
            "Std(Torr)": mean_std
        })

        # Save to file
        filename = f"{sample_name}_total_outgassing.txt"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, sep='\t', index=False, float_format="%.6e")
        print(f"Saved mean trace: {filepath}")

def save_grouped_mass_spectra(data_dict, output_dir, sample_groups):
    """
    Group samples using a predefined dictionary, average their spectra,
    and save to '_averaged' folder.

    Parameters:
    - data_dict: dict of {sample_name: {'avg': [...], 'std': [...]}}
    - output_dir: original output folder (used to derive '_averaged')
    - sample_groups: dict of {group_name: [sample_name1, sample_name2, ...]}
    """
    averaged_dir = output_dir

    for group_name, sample_list in sample_groups.items():
        samples = []
        for sample_name in sample_list:
            if sample_name not in data_dict:
                print(f"⚠️ Warning: Sample '{sample_name}' not found in data_dict.")
                continue
            samples.append(data_dict[sample_name])

        if not samples:
            print(f"⚠️ Skipping group '{group_name}' — no valid samples found.")
            continue

        # Validate all samples have same length
        lengths = [len(s['avg']) for s in samples]
        if len(set(lengths)) != 1:
            raise ValueError(f"Samples in group '{group_name}' have inconsistent lengths: {lengths}")

        # Stack and average
        avg_stack = np.array([s['avg'] for s in samples])
        std_stack = np.array([s['std'] for s in samples])

        avg_mean = np.mean(avg_stack, axis=0)
        std_mean = np.std(avg_stack, axis=0)

        df = pd.DataFrame({
            "MZ": range(1, len(avg_mean) + 1),
            "Pressure(Torr)": avg_mean,
            "Std(Torr)": std_mean
        })

        filename = f"{group_name}_MS_averaged.txt"
        filepath = os.path.join(averaged_dir, filename)
        df.to_csv(filepath, sep='\t', index=False, float_format="%.6e")
        print(f"✅ Saved averaged spectrum: {filepath}")


def save_grouped_sample_ion_to_txt(sample_ion, output_dir, sample_groups):
    """
    Group samples using a predefined dictionary, average their ion spectra over time,
    and save to '_averaged' folder.

    Parameters:
    - sample_ion: dict of {sample_name: {mz: (time[], corrected[], std[])}}
    - output_dir: folder to save averaged files
    - sample_groups: dict of {group_name: [sample_name1, sample_name2, ...]}
    """
    os.makedirs(output_dir, exist_ok=True)

    for group_name, sample_list in sample_groups.items():
        mz_dicts = []
        for sample_name in sample_list:
            if sample_name not in sample_ion:
                print(f"⚠️ Warning: Sample '{sample_name}' not found in sample_ion.")
                continue
            mz_dicts.append(sample_ion[sample_name])

        if not mz_dicts:
            print(f"⚠️ Skipping group '{group_name}' — no valid samples found.")
            continue

        # Merge all mz_dicts in the group
        merged_mz = defaultdict(list)
        for mz_dict in mz_dicts:
            filtered = {k: v for k, v in mz_dict.items() if k != 'sum_std'}
            for mz, (time, corrected, std) in filtered.items():
                merged_mz[mz].append((np.array(time), np.array(corrected), np.array(std)))

        # Assume all time arrays are identical; use the first one
        first_mz = next(iter(merged_mz))
        time_array = merged_mz[first_mz][0][0]
        columns = [pd.Series(time_array, name="Time(s)")]

        # Average across samples for each m/z
        for mz in sorted(merged_mz.keys()):
            time_stack = np.stack([entry[0] for entry in merged_mz[mz]])
            corrected_stack = np.stack([entry[1] for entry in merged_mz[mz]])
            std_stack = np.stack([entry[2] for entry in merged_mz[mz]])

            if not (time_stack.shape == corrected_stack.shape == std_stack.shape):
                raise ValueError(f"Shape mismatch in group '{group_name}', m/z {mz}")

            corrected_avg = np.mean(corrected_stack, axis=0)
            std_avg = np.std(corrected_stack, axis=0)

            columns.append(pd.Series(corrected_avg, name=f"MZ{mz}(Torr)"))
            columns.append(pd.Series(std_avg, name=f"Std{mz}(Torr)"))

        # Concatenate all columns
        df = pd.concat(columns, axis=1)

        # Save to file
        filename = f"{group_name}_MS_t_averaged.txt"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, sep='\t', index=False, float_format="%.6e")
        print(f"✅ Saved averaged file: {filepath}")

def save_grouped_sample_ion_to_total_outgassing_txt(sample_ion, output_dir, sample_groups):
    """
    Group samples using a predefined dictionary, sum their ion spectra across m/z,
    and save total outgassing traces to '_averaged' folder.

    Parameters:
    - sample_ion: dict of {sample_name: {mz: (time[], corrected[], std[]), 'sum_std': [...]}}
    - output_dir: folder to save averaged files
    - sample_groups: dict of {group_name: [sample_name1, sample_name2, ...]}
    """
    os.makedirs(output_dir, exist_ok=True)

    for group_name, sample_list in sample_groups.items():
        mz_accumulator = []
        std_accumulator = []

        valid_entries = []
        for sample_name in sample_list:
            if sample_name not in sample_ion:
                print(f"⚠️ Warning: Sample '{sample_name}' not found in sample_ion.")
                continue
            valid_entries.append((sample_name, sample_ion[sample_name]))

        if not valid_entries:
            print(f"⚠️ Skipping group '{group_name}' — no valid samples found.")
            continue

        # Assume all time arrays are identical; use the first one
        first_sample_name, first_mz_dict = valid_entries[0]
        first_mz = next(iter(first_mz_dict))
        time_array = first_mz_dict[first_mz][0]

        for sample_name, mz_dict in valid_entries:
            mz_values = []
            for mz in range(1, 201):  # Loop over mz1 to mz200
                if mz in mz_dict:
                    time, corrected, std = mz_dict[mz]
                    if not (len(time) == len(corrected) == len(std)):
                        raise ValueError(f"Length mismatch in sample '{sample_name}', m/z {mz}")
                    mz_values.append(corrected)

            # Sum across m/z channels for this sample
            mz_array = np.array(mz_values)  # shape: (num_mz, num_time)
            summed_trace = np.sum(mz_array, axis=0)  # shape: (num_time,)
            mz_accumulator.append(summed_trace)

            # Collect std arrays (assumed precomputed as 'sum_std')
            std_accumulator.append(np.array(mz_dict['sum_std']))

        # Stack and compute mean/std across samples
        mz_stack = np.stack(mz_accumulator)  # shape: (num_samples, num_time)
        summed_mz = np.mean(mz_stack, axis=0)  # shape: (num_time,)
        mean_std = np.std(mz_stack, axis=0, ddof=1)  # shape: (num_time,)

        # Build DataFrame
        df = pd.DataFrame({
            "Time(s)": [f"{t:.6f}".rstrip('0').rstrip('.') for t in time_array],
            "Averaged_Pressure(Torr)": summed_mz,
            "Std(Torr)": mean_std
        })

        # Save to file
        filename = f"{group_name}_total_outgassing_averaged.txt"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, sep='\t', index=False, float_format="%.6e")
        print(f"✅ Saved mean trace: {filepath}")

def plot_ascii_files(input_folder, output_folder, extensions=(".txt", ".dat")):
    # Create output folder if it doesn’t exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop through files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(extensions):
            filepath = os.path.join(input_folder, filename)

            try:
                # Read file with pandas (handles tab/space delimiters)
                df = pd.read_csv(filepath, sep="\t")

                # Extract column names
                cols = df.columns.tolist()

                if len(cols) < 2:
                    print(f"Skipping {filename}, not enough columns")
                    continue

                x = df[cols[0]]
                y = df[cols[1]]

                plt.figure(figsize=(6,4))

                if len(cols) == 2:
                    plt.plot(x, y, linestyle="-", label=cols[1])
                elif len(cols) >= 3:
                    std = df[cols[2]]
                    plt.plot(x, y, color="blue", label=cols[1])
                    plt.fill_between(x, y-std, y+std, color="blue", alpha=0.3,
                                     label=f"{cols[1]} ± {cols[2]}")

                plt.xlabel(cols[0])
                plt.ylabel(cols[1])
                plt.title(filename)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()

                # Save plot in output folder
                outpath = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
                plt.savefig(outpath, dpi=150)
                plt.close()
                print(f"Saved plot: {outpath}")

            except Exception as e:
                print(f"Could not process {filename}: {e}")

def plot_MS(sample_outgassing,output):
    os.makedirs(output, exist_ok=True)
    for key in sample_outgassing.keys():
        plt.figure(figsize=(8, 6))
        
        outgassing_avg = sample_outgassing[key]['avg']
        #print(np.shape(outgassing_avg))
        outgassing_std = sample_outgassing[key]['std']
        #ncols = outgassing_avg.shape[0]

        outgassing_avg[outgassing_avg < 0] = 0
        outgassing_std[outgassing_std > outgassing_avg] = 0

        plt.bar(np.arange(1,len(outgassing_avg)+1),outgassing_avg, alpha=0.5, color='gray', edgecolor='black', label=key)
        plt.errorbar(np.arange(1,len(outgassing_avg)+1),outgassing_avg, yerr=outgassing_std, fmt='none', capsize=3, ecolor='black')

        for x_val in range(5, len(outgassing_avg)+1, 5):
            plt.axvline(x=x_val, color='gray', linestyle='--', alpha=0.7, linewidth=0.5)

        plt.xlabel("m/z")
        plt.ylabel("Pressure increase (Torr)")
        plt.yscale('log')
        plt.ylim(bottom=1e-12)
        plt.title(f"{key} Outgassing Spectrum (Log Scale)")
        plt.grid(True)
        plt.tight_layout()
        outpath = os.path.join(output, f"{key}_MS_log.png")
        plt.savefig(outpath, dpi=150)
        plt.close()

def plot_MS_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.endswith(".txt"):
            filepath = os.path.join(input_folder, fname)
            try:
                df = pd.read_csv(filepath, sep="\t")

                # Extract sample name from filename (without extension)
                sample_name = os.path.splitext(fname)[0]

                # Extract data
                mz = df["MZ"].values
                avg = df["Pressure(Torr)"].values
                std = df["Std(Torr)"].values

                # Clean data
                avg[avg < 0] = 0
                std[std > avg] = 0

                # Plot
                plt.figure(figsize=(8, 6))
                plt.bar(mz, avg, alpha=0.5, color='gray', edgecolor='black', label=sample_name)
                plt.errorbar(mz, avg, yerr=std, fmt='none', capsize=3, ecolor='black')

                for x_val in range(5, int(max(mz)) + 1, 5):
                    plt.axvline(x=x_val, color='gray', linestyle='--', alpha=0.7, linewidth=0.5)

                plt.xlabel("m/z")
                plt.ylabel("Pressure increase (Torr)")
                plt.yscale('log')
                plt.title(f"{sample_name} Outgassing Spectrum (Log Scale)")
                plt.grid(True)
                plt.ylim(bottom=1e-12)
                plt.tight_layout()

                outpath = os.path.join(output_folder, f"{sample_name}_log.png")
                plt.savefig(outpath, dpi=150)
                plt.close()
                print(f"Saved plot: {outpath}")

            except Exception as e:
                print(f"Failed to process {fname}: {e}")

def plot_total_outgassing_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.endswith(".txt"):
            filepath = os.path.join(input_folder, fname)
            try:
                df = pd.read_csv(filepath, sep="\t")

                # Extract sample name from filename (without extension)
                sample_name = os.path.splitext(fname)[0]
                # Extract data
                time_data = df["Time(s)"].values
                intensity_avg = df["Averaged_Pressure(Torr)"].values
                intensity_std = df["Std(Torr)"].values

                # Plot
                plt.figure(figsize=(8, 5))
                plt.plot(time_data, intensity_avg, label='Average Intensity', color='blue')
                plt.fill_between(time_data, intensity_avg - intensity_std, intensity_avg + intensity_std,
                                color='lightblue', alpha=0.5, label='±1 Std Dev')
                plt.xlabel('Time (s)')
                plt.ylabel('Intensity')
                plt.title(f'Averaged Signal for {sample_name}')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

                outpath = os.path.join(output_folder, f"{sample_name}.png")
                plt.savefig(outpath, dpi=150)
                plt.close()
                print(f"Saved plot: {outpath}")

            except Exception as e:
                print(f"Failed to process {fname}: {e}")

def plot_MS_t_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if fname.endswith(".txt"):
            filepath = os.path.join(input_folder, fname)
            try:
                df = pd.read_csv(filepath, sep="\t")

                # Extract sample name from filename (without extension)
                sample_name = os.path.splitext(fname)[0]
                # Extract data
                # Extract MZ{i} columns and sort
                mz_cols = [col for col in df.columns if col.startswith("MZ")]
                mz_indices = sorted([int(col[2:-6]) for col in mz_cols]) #if 1 <= int(col[2:-6]) <= 100
                mz_matrix = np.array([df[f"MZ{i}(Torr)"] for i in mz_indices])
                time = df['Time(s)'].values

                # Compute sums
                sum_over_mz = mz_matrix.sum(axis=0)  # shape: (len(time),)
                sum_over_time = mz_matrix.sum(axis=1)  # shape: (len(mz_indices),)

                # Plot
                # Create layout
                fig = plt.figure(figsize=(12, 8))
                gs = GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], hspace=0.05, wspace=0.05)

                ax_main = fig.add_subplot(gs[1, 0])
                ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
                ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)

                # Main colormap
                im = ax_main.imshow(
                    np.abs(mz_matrix),
                    aspect='auto',
                    extent=[time.min(), time.max(), mz_indices[0], mz_indices[-1]],
                    origin='lower',
                    cmap='viridis',
                    norm=LogNorm(vmin=max(mz_matrix.min(), 1e-12), vmax=mz_matrix.max())
                )
                ax_main.set_xlabel("Time(s)")
                ax_main.set_ylabel("m/z")
                #ax_main.set_title(f"Colormap — {label}")

                ymin, ymax = ax_main.get_ylim()

                # Add vertical lines
                #ax_main.axvline(x=50, color='red', linestyle='--', linewidth=3)
                #ax_main.axvline(x=300, color='red', linestyle='--', linewidth=3)


                # Top plot: sum over MZ
                ax_top.plot(time, sum_over_mz, color='black')
                ax_top.set_ylabel("Total Pressure (Torr)", labelpad=15)
                ax_top.tick_params(labelbottom=False)

                # Side plot: sum over time
                ax_side.barh(mz_indices, sum_over_time, color='black')
                ax_side.set_xlabel("Pressure (Torr)", labelpad=15)
                ax_side.tick_params(labelleft=False)
                #ax_side.set_xscale('log')

                ax_main.set_ylim(ymin, ymax)
                ax_side.set_ylim(ymin, ymax)

                # Colorbar
                cbar = fig.colorbar(im, ax=[ax_main, ax_top, ax_side], orientation='vertical', pad=0.02)
                cbar.set_label("Pressure (Torr)")

                #plt.tight_layout()
                #fig.subplots_adjust(top=0.88)  # Add extra space for title
                fig.suptitle(f"Colormap — {sample_name}", fontsize=16)  # Title for entire figure

                outpath = os.path.join(output_folder, f"{sample_name}.png")
                fig.savefig(outpath, dpi=150)
                plt.close()
                print(f"Saved plot: {outpath}")

            except Exception as e:
                print(f"Failed to process {fname}: {e}")