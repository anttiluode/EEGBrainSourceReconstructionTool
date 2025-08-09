# EEG Brain Source Reconstruction Tool (v4 - Realistic Terminology)
#
# This version uses proper neuroscience terminology while maintaining
# the same robust functionality for EEG source localization.

import os
import tkinter as tk
from tkinter import filedialog, messagebox
import mne
import numpy as np
from scipy.signal import hilbert
import logging

# Suppress verbose MNE logging for a cleaner output
logging.basicConfig(level=logging.WARNING)

def check_and_download_fsaverage():
    """Checks for MNE's fsaverage dataset and downloads it if missing."""
    print("Checking for MNE fsaverage brain template...")
    
    # Try to get the subjects directory - handle different MNE versions
    try:
        # Method 1: Try the old way first
        subjects_dir = mne.datasets.get_subjects_dir()
    except AttributeError:
        try:
            # Method 2: Try the newer way
            subjects_dir = mne.get_subjects_dir()
        except AttributeError:
            # Method 3: Use environment variable or default
            subjects_dir = os.environ.get('SUBJECTS_DIR')
            if subjects_dir is None:
                subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data', 'MNE-fsaverage-data')

    if subjects_dir is None or not os.path.exists(subjects_dir):
        # Create a default subjects directory
        subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data', 'MNE-fsaverage-data')
        os.makedirs(subjects_dir, exist_ok=True)
        print(f"Using subjects directory: {subjects_dir}")

    fsaverage_path = os.path.join(subjects_dir, 'fsaverage')
    if not os.path.exists(fsaverage_path):
        print("fsaverage brain template not found. Downloading...")
        try:
            # Try to download fsaverage
            mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
        except Exception as e:
            print(f"Download failed: {e}")
            # Alternative: just create a basic structure
            os.makedirs(fsaverage_path, exist_ok=True)
    
    print(f"fsaverage brain template location: {fsaverage_path}")
    return subjects_dir

def create_inverse_solution(raw, subjects_dir):
    """
    Creates the MNE inverse solution for EEG source reconstruction.
    Uses sLORETA method to estimate cortical source activity from scalp EEG.
    """
    print("Creating head model (BEM)...")
    # Use 3-layer BEM model for EEG: brain, skull, scalp
    model = mne.make_bem_model(subject='fsaverage', ico=4, 
                               conductivity=(0.3, 0.006, 0.3),  # brain, skull, scalp
                               subjects_dir=subjects_dir, verbose=False)
    bem_sol = mne.make_bem_solution(model, verbose=False)
    
    print("Setting up cortical source space...")
    src = mne.setup_source_space('fsaverage', spacing='ico5', add_dist=False, 
                                 subjects_dir=subjects_dir, verbose=False)
    
    print("Computing forward solution...")
    fwd = mne.make_forward_solution(raw.info, trans='fsaverage', src=src, bem=bem_sol, 
                                    meg=False, eeg=True, mindist=5.0, verbose=False)
    
    print("Computing noise covariance...")
    noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
    
    print("Creating inverse operator (sLORETA)...")
    inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, 
                                                              loose=0.2, depth=0.8, verbose=False)
    return inverse_operator

class EEGSourceReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Brain Source Reconstruction")
        self.root.geometry("550x650")
        
        # Make window resizable
        self.root.resizable(True, True)

        # Title and description
        title_label = tk.Label(root, text="EEG Brain Source Reconstruction", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        description = ("Load multi-channel EEG data to reconstruct brain source activity.\n"
                      "This tool uses sLORETA inverse modeling to estimate where in the brain\n"
                      "the recorded EEG signals likely originated from.")
        self.desc_label = tk.Label(root, text=description, wraplength=450, justify="center")
        self.desc_label.pack(pady=5)

        # Analysis options
        options_frame = tk.LabelFrame(root, text="Analysis Options", font=("Arial", 10, "bold"))
        options_frame.pack(pady=15, padx=20, fill="both", expand=True)

        # Frequency band selection
        freq_frame = tk.Frame(options_frame)
        freq_frame.pack(pady=10)
        
        tk.Label(freq_frame, text="Frequency Band:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.freq_var = tk.StringVar(value="theta")
        freq_options = [
            ("Delta (0.5-4 Hz) - Deep sleep, unconscious processes", "delta"),
            ("Theta (4-8 Hz) - Memory, meditation, creativity", "theta"), 
            ("Alpha (8-12 Hz) - Relaxed awareness, eyes closed", "alpha"),
            ("Beta (12-30 Hz) - Alert, focused attention", "beta"),
            ("Gamma (30-50 Hz) - Binding, consciousness", "gamma")
        ]
        
        for text, value in freq_options:
            tk.Radiobutton(freq_frame, text=text, variable=self.freq_var, 
                          value=value, font=("Arial", 9)).pack(anchor="w", padx=20)

        # Analysis type
        analysis_frame = tk.Frame(options_frame)
        analysis_frame.pack(pady=10)
        
        tk.Label(analysis_frame, text="Visualization Type:", font=("Arial", 10, "bold")).pack(anchor="w")
        
        self.analysis_var = tk.StringVar(value="phase")
        analysis_options = [
            ("Phase Patterns - Shows timing/synchronization of brain waves", "phase"),
            ("Power Distribution - Shows strength of brain activity", "power"),
            ("Raw Signal Amplitude - Shows direct reconstructed signals", "raw")
        ]
        
        for text, value in analysis_options:
            tk.Radiobutton(analysis_frame, text=text, variable=self.analysis_var, 
                          value=value, font=("Arial", 9)).pack(anchor="w", padx=20)

        # Action button
        self.load_button = tk.Button(root, text="Load EEG File & Reconstruct Brain Sources", 
                                     command=self.run_reconstruction,
                                     bg="#4CAF50", fg="white", font=("Arial", 12))
        self.load_button.pack(pady=20)
        
        # Status labels
        self.status_label = tk.Label(root, text="Ready to load EEG file...", wraplength=450)
        self.status_label.pack(pady=5)

        self.progress_label = tk.Label(root, text="", fg="blue", wraplength=450)
        self.progress_label.pack(pady=5)

        # Info section
        info_text = ("ℹ️ This tool performs standard EEG source localization using MNE-Python.\n"
                    "It estimates brain activity from scalp recordings using the sLORETA method.\n"
                    "Results show likely cortical sources of your EEG signals.")
        self.info_label = tk.Label(root, text=info_text, wraplength=450, 
                                  font=("Arial", 8), fg="gray")
        self.info_label.pack(pady=10)

    def update_status(self, message):
        """Helper to update status and refresh GUI"""
        self.status_label.config(text=message)
        self.root.update()
        print(message)

    def get_frequency_band(self):
        """Get frequency range based on selected band"""
        freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 12), 
            "beta": (12, 30),
            "gamma": (30, 50)
        }
        return freq_bands[self.freq_var.get()]

    def run_reconstruction(self):
        filepath = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[("EEG Files", "*.edf *.bdf *.fif"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            self.update_status("Step 1/8: Setting up brain model...")
            subjects_dir = check_and_download_fsaverage()

            self.update_status("Step 2/8: Loading EEG data...")
            raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            raw.pick_types(eeg=True)
            
            # Show file info
            n_channels = len(raw.ch_names)
            duration = raw.times[-1]
            sfreq = raw.info['sfreq']
            filename = os.path.basename(filepath)
            print(f"Loaded: {filename} ({n_channels} channels, {duration:.1f}s, {sfreq:.0f} Hz)")
            
            self.update_status("Step 3/8: Preprocessing EEG data...")
            # Only resample if needed
            if raw.info['sfreq'] > 150:
                raw.resample(100, verbose=False)
            
            # Clean channel names
            cleaned_names = {name: name.strip().replace('.', '').upper() for name in raw.ch_names}
            raw.rename_channels(cleaned_names, verbose=False)
            
            # Apply montage and drop channels without positions
            montage = mne.channels.make_standard_montage('standard_1005')
            raw.set_montage(montage, on_missing='warn', verbose=False)
            
            # Drop channels that don't have 3D positions
            pos_dict = raw.get_montage().get_positions()['ch_pos']
            channels_with_pos = [ch for ch in raw.ch_names 
                               if ch in pos_dict and np.isfinite(pos_dict[ch]).all()]
            
            if len(channels_with_pos) < len(raw.ch_names):
                dropped = set(raw.ch_names) - set(channels_with_pos)
                print(f"Dropping {len(dropped)} channels without positions: {sorted(dropped)}")
                raw.pick(channels_with_pos, verbose=False)
            
            if len(channels_with_pos) < 4:
                raise ValueError(f"Only {len(channels_with_pos)} channels have valid positions. Need at least 4.")
            
            print(f"Using {len(channels_with_pos)} channels with valid 3D positions")
            raw.set_eeg_reference('average', projection=True, verbose=False)
            
            # Get selected frequency band
            low_freq, high_freq = self.get_frequency_band()
            band_name = self.freq_var.get().title()
            
            self.update_status(f"Step 4/8: Filtering for {band_name} band ({low_freq}-{high_freq} Hz)...")
            raw.filter(low_freq, high_freq, fir_design='firwin', verbose=False)

            self.update_status("Step 5/8: Creating inverse solution (this may take time)...")
            inverse_operator = create_inverse_solution(raw, subjects_dir)

            self.update_status("Step 6/8: Applying inverse solution to reconstruct sources...")
            # Process only a subset of data for speed
            tmax = min(10.0, raw.times[-1])  # Only process first 10 seconds
            raw_cropped = raw.copy().crop(tmax=tmax)
            stc = mne.minimum_norm.apply_inverse_raw(raw_cropped, inverse_operator, 
                                                     lambda2=1.0 / 9.0, method='sLORETA', verbose=False)

            # Apply selected analysis
            analysis_type = self.analysis_var.get()
            
            if analysis_type == "phase":
                self.update_status("Step 7/8: Calculating instantaneous phase patterns...")
                # Calculate instantaneous phase using Hilbert transform
                phase_data = np.angle(hilbert(stc.data))
                stc_final = mne.SourceEstimate(phase_data, vertices=stc.vertices, 
                                             tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')
                colormap = 'twilight_shifted'
                clim = dict(kind='value', lims=[-np.pi, 0, np.pi])
                viz_title = f"Brain Phase Patterns - {band_name} Band"
                
            elif analysis_type == "power":
                self.update_status("Step 7/8: Calculating power distribution...")
                # Calculate power (amplitude squared)
                power_data = stc.data ** 2
                stc_final = mne.SourceEstimate(power_data, vertices=stc.vertices,
                                             tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage')
                colormap = 'hot'
                clim = dict(kind='percent', lims=[70, 90, 99])
                viz_title = f"Brain Power Distribution - {band_name} Band"
                
            else:  # raw
                self.update_status("Step 7/8: Preparing raw signal visualization...")
                stc_final = stc
                colormap = 'RdBu_r'
                clim = dict(kind='percent', lims=[5, 50, 95])
                viz_title = f"Brain Source Signals - {band_name} Band"

            self.update_status("Step 8/8: Creating 3D brain visualization...")
            
            # Try different backends for 3D plotting
            backends_to_try = ['pyvistaqt', 'pyvista', 'mayavi']
            brain = None
            
            for backend in backends_to_try:
                try:
                    print(f"Trying 3D backend: {backend}")
                    mne.viz.set_3d_backend(backend)
                    
                    # Create appropriate time label
                    if analysis_type == "phase":
                        time_label = 'Phase Patterns (t=%0.2f s)'
                    elif analysis_type == "power":
                        time_label = 'Power Distribution (t=%0.2f s)'
                    else:
                        time_label = 'Source Activity (t=%0.2f s)'
                    
                    # Plot the 3D brain
                    brain = stc_final.plot(
                        subjects_dir=subjects_dir,
                        subject='fsaverage',
                        surface='pial',
                        hemi='both',
                        colormap=colormap,
                        time_label=time_label,
                        backend=backend,
                        smoothing_steps=5,
                        size=(1200, 800),
                        clim=clim
                    )
                    brain.add_text(0.1, 0.9, viz_title, 'title', font_size=16)
                    
                    # Add educational info
                    info_text = f"sLORETA source reconstruction | {len(channels_with_pos)} EEG channels"
                    brain.add_text(0.1, 0.05, info_text, 'title', font_size=10)
                    
                    print(f"Success with backend: {backend}")
                    break
                    
                except Exception as e:
                    print(f"Backend {backend} failed: {e}")
                    continue
            
            if brain is None:
                # Fallback: create 2D plots instead
                print("All 3D backends failed. Creating 2D plots instead...")
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Plot some time points as activity maps
                times_to_plot = [0.0, 2.0, 4.0, 6.0]
                for i, t in enumerate(times_to_plot):
                    if i < 4 and t <= stc_final.times[-1]:
                        ax = axes[i//2, i%2]
                        
                        # Simple time series plot for each subplot
                        time_idx = np.argmin(np.abs(stc_final.times - t))
                        data_at_time = stc_final.data[:, time_idx]
                        
                        ax.hist(data_at_time, bins=30, alpha=0.7)
                        ax.set_title(f'{analysis_type.title()} at t={t:.1f}s')
                        ax.set_xlabel(f'{analysis_type.title()} Value')
                        ax.set_ylabel('Number of Sources')
                
                plt.suptitle(f'{viz_title}\n(3D visualization unavailable)', fontsize=14)
                plt.tight_layout()
                plt.show()
                
                self.update_status("✓ Done! 2D analysis plots created (3D backend unavailable)")
            else:
                self.update_status("✓ Done! Interactive 3D brain visualization is now open.")
                
            self.progress_label.config(text="You can rotate, zoom, and animate through time in the 3D window!", 
                                       fg="green")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.update_status(error_msg)
            messagebox.showerror("Error", f"An error occurred:\n\n{e}\n\nCheck console for details.")
            logging.exception("Error during source reconstruction")

if __name__ == "__main__":
    # Try to set the 3D backend, but don't fail if it doesn't work
    print("EEG Brain Source Reconstruction Tool")
    print("Setting up 3D visualization backend...")
    
    try:
        import pyvistaqt
        mne.viz.set_3d_backend("pyvistaqt")
        print("✓ Using pyvistaqt backend for 3D visualization")
    except ImportError:
        try:
            import pyvista
            mne.viz.set_3d_backend("pyvista")
            print("✓ Using pyvista backend for 3D visualization")
        except ImportError:
            try:
                import mayavi
                mne.viz.set_3d_backend("mayavi")
                print("✓ Using mayavi backend for 3D visualization")
            except ImportError:
                print("⚠ No 3D backends available. Will use 2D fallback plots.")
                print("  Install with: pip install pyvista pyvistaqt")
    except Exception as e:
        print(f"Backend setup warning: {e}")
    
    print("\nStarting application...")
    root = tk.Tk()
    app = EEGSourceReconstructionApp(root)
    root.mainloop()