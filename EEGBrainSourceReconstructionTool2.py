# EEG Brain Source Reconstruction Tool (Enhanced Version)
#
# Improved version with better preprocessing, multiple inverse methods,
# and enhanced visualization options

import os
import sys
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mne
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import threading
import time

# Suppress verbose MNE logging and Qt warnings
logging.basicConfig(level=logging.WARNING)
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', message='.*QApplication.*')
warnings.filterwarnings('ignore', message='.*QWindowsWindow.*')

class SourceReconstructor:
    """Handles different source reconstruction methods"""
    
    def __init__(self):
        self.methods = {
            'sLORETA': {'method': 'sLORETA', 'lambda2': 1.0 / 9.0},
            'dSPM': {'method': 'dSPM', 'lambda2': 1.0 / 9.0},
            'MNE': {'method': 'MNE', 'lambda2': 1.0 / 9.0},
            'eLORETA': {'method': 'eLORETA', 'lambda2': 1.0 / 9.0}
        }
    
    def reconstruct(self, raw, inverse_operator, method='sLORETA'):
        """Apply inverse solution with specified method"""
        params = self.methods[method]
        stc = mne.minimum_norm.apply_inverse_raw(
            raw, inverse_operator,
            lambda2=params['lambda2'],
            method=params['method'],
            verbose=False
        )
        return stc

class PreprocessingPipeline:
    """Handles EEG preprocessing steps"""
    
    @staticmethod
    def detect_bad_channels(raw, threshold=3.0):
        """Simple bad channel detection based on variance"""
        data = raw.get_data()
        channel_vars = np.var(data, axis=1)
        z_scores = np.abs((channel_vars - np.median(channel_vars)) / np.std(channel_vars))
        bad_channels = [raw.ch_names[i] for i in np.where(z_scores > threshold)[0]]
        return bad_channels
    
    @staticmethod
    def remove_artifacts(raw, method='basic'):
        """Remove common artifacts"""
        if method == 'basic':
            # Simple high-pass filter to remove drift
            raw.filter(l_freq=0.5, h_freq=None, fir_design='firwin', verbose=False)
            
            # Notch filter for line noise - check power line frequency
            # Europe = 50 Hz, Americas = 60 Hz
            line_freq = 50  # Default to European standard
            
            # Apply notch filter for line noise and harmonics
            # Only apply to frequencies below Nyquist
            nyquist = raw.info['sfreq'] / 2.0
            freqs = []
            for harmonic in range(1, 5):  # First 4 harmonics
                freq = line_freq * harmonic
                if freq < nyquist - 1:  # Leave 1 Hz margin from Nyquist
                    freqs.append(freq)
            
            if freqs:
                raw.notch_filter(freqs, fir_design='firwin', verbose=False)
            
        elif method == 'ica':
            # ICA-based artifact removal (simplified)
            try:
                from mne.preprocessing import ICA
                ica = ICA(n_components=min(15, len(raw.ch_names)-1), random_state=42)
                ica.fit(raw, verbose=False)
                
                # Find EOG artifacts
                eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
                ica.exclude = eog_indices[:2]  # Remove top 2 EOG components
                raw = ica.apply(raw, verbose=False)
            except:
                # Fallback to basic if ICA fails
                PreprocessingPipeline.remove_artifacts(raw, 'basic')
        
        return raw

class EEGSourceReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Brain Source Reconstruction - Enhanced")
        self.root.geometry("700x850")
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Initialize components
        self.reconstructor = SourceReconstructor()
        self.preprocessing = PreprocessingPipeline()
        self.processing_thread = None
        self.brain_figures = []  # Keep track of brain figures
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the user interface"""
        # Title
        title_label = tk.Label(self.root, text="EEG Brain Source Reconstruction", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Basic tab
        self.basic_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.basic_frame, text="Basic Settings")
        self.create_basic_tab()
        
        # Advanced tab
        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced Settings")
        self.create_advanced_tab()
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.create_results_tab()
        
        # Action buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.load_button = tk.Button(button_frame, text="Load EEG File", 
                                     command=self.load_file,
                                     bg="#2196F3", fg="white", font=("Arial", 10))
        self.load_button.pack(side=tk.LEFT, padx=5)
        
        self.process_button = tk.Button(button_frame, text="Process & Reconstruct", 
                                        command=self.run_reconstruction,
                                        bg="#4CAF50", fg="white", font=("Arial", 10),
                                        state=tk.DISABLED)
        self.process_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop", 
                                     command=self.stop_processing,
                                     bg="#f44336", fg="white", font=("Arial", 10),
                                     state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.close_brain_button = tk.Button(button_frame, text="Close 3D Views", 
                                           command=self.close_brain_views,
                                           bg="#FF9800", fg="white", font=("Arial", 10))
        self.close_brain_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(fill=tk.X, padx=20, pady=5)
        
        # Status
        self.status_label = tk.Label(self.root, text="Ready to load EEG file...", 
                                     wraplength=650)
        self.status_label.pack(pady=5)
        
        # File info
        self.file_info_label = tk.Label(self.root, text="", fg="blue", wraplength=650)
        self.file_info_label.pack(pady=5)
        
    def create_basic_tab(self):
        """Create basic settings tab"""
        # Frequency band selection
        freq_frame = tk.LabelFrame(self.basic_frame, text="Frequency Band", 
                                   font=("Arial", 10, "bold"))
        freq_frame.pack(pady=10, padx=20, fill="x")
        
        self.freq_var = tk.StringVar(value="alpha")
        freq_options = [
            ("Delta (0.5-4 Hz) - Deep sleep", "delta"),
            ("Theta (4-8 Hz) - Memory, meditation", "theta"), 
            ("Alpha (8-12 Hz) - Relaxed awareness", "alpha"),
            ("Beta (12-30 Hz) - Active thinking", "beta"),
            ("Gamma (30-50 Hz) - Cognitive processing", "gamma"),
            ("Broadband (0.5-50 Hz) - All frequencies", "broadband")
        ]
        
        for i, (text, value) in enumerate(freq_options):
            tk.Radiobutton(freq_frame, text=text, variable=self.freq_var, 
                          value=value, font=("Arial", 9)).grid(row=i//2, column=i%2, 
                                                               sticky="w", padx=10, pady=2)
        
        # Visualization type
        viz_frame = tk.LabelFrame(self.basic_frame, text="Visualization Type", 
                                  font=("Arial", 10, "bold"))
        viz_frame.pack(pady=10, padx=20, fill="x")
        
        self.viz_var = tk.StringVar(value="power")
        viz_options = [
            ("Power Distribution - Shows strength of activity", "power"),
            ("Phase Patterns - Shows timing/synchronization", "phase"),
            ("Raw Amplitude - Shows direct signals", "raw"),
            ("Statistical Map - Shows significant activations", "stats")
        ]
        
        for i, (text, value) in enumerate(viz_options):
            tk.Radiobutton(viz_frame, text=text, variable=self.viz_var, 
                          value=value, font=("Arial", 9)).pack(anchor="w", padx=10)
        
        # Time window
        time_frame = tk.LabelFrame(self.basic_frame, text="Time Window", 
                                   font=("Arial", 10, "bold"))
        time_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(time_frame, text="Analyze from:").grid(row=0, column=0, padx=5, pady=5)
        self.time_start_var = tk.DoubleVar(value=0.0)
        self.time_start_spin = tk.Spinbox(time_frame, from_=0, to=1000, increment=0.5,
                                          textvariable=self.time_start_var, width=10)
        self.time_start_spin.grid(row=0, column=1, padx=5, pady=5)
        tk.Label(time_frame, text="to").grid(row=0, column=2, padx=5, pady=5)
        
        self.time_end_var = tk.DoubleVar(value=10.0)
        self.time_end_spin = tk.Spinbox(time_frame, from_=0, to=1000, increment=0.5,
                                        textvariable=self.time_end_var, width=10)
        self.time_end_spin.grid(row=0, column=3, padx=5, pady=5)
        tk.Label(time_frame, text="seconds").grid(row=0, column=4, padx=5, pady=5)
        
    def create_advanced_tab(self):
        """Create advanced settings tab"""
        # Preprocessing options
        preproc_frame = tk.LabelFrame(self.advanced_frame, text="Preprocessing", 
                                      font=("Arial", 10, "bold"))
        preproc_frame.pack(pady=10, padx=20, fill="x")
        
        self.remove_bad_channels_var = tk.BooleanVar(value=True)
        tk.Checkbutton(preproc_frame, text="Automatically detect and remove bad channels",
                       variable=self.remove_bad_channels_var).pack(anchor="w", padx=10, pady=2)
        
        self.artifact_removal_var = tk.StringVar(value="basic")
        tk.Label(preproc_frame, text="Artifact removal:").pack(anchor="w", padx=10, pady=2)
        tk.Radiobutton(preproc_frame, text="Basic (filters only)", 
                       variable=self.artifact_removal_var, value="basic").pack(anchor="w", padx=30)
        tk.Radiobutton(preproc_frame, text="ICA-based (removes EOG artifacts)", 
                       variable=self.artifact_removal_var, value="ica").pack(anchor="w", padx=30)
        
        # Source reconstruction options
        source_frame = tk.LabelFrame(self.advanced_frame, text="Source Reconstruction", 
                                     font=("Arial", 10, "bold"))
        source_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(source_frame, text="Inverse method:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.method_var = tk.StringVar(value="sLORETA")
        method_menu = ttk.Combobox(source_frame, textvariable=self.method_var,
                                   values=["sLORETA", "dSPM", "MNE", "eLORETA"],
                                   state="readonly", width=15)
        method_menu.grid(row=0, column=1, padx=10, pady=5)
        
        tk.Label(source_frame, text="Source space:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.spacing_var = tk.StringVar(value="ico5")
        spacing_menu = ttk.Combobox(source_frame, textvariable=self.spacing_var,
                                    values=["ico4", "ico5", "oct5", "oct6"],
                                    state="readonly", width=15)
        spacing_menu.grid(row=1, column=1, padx=10, pady=5)
        
        # Regularization
        tk.Label(source_frame, text="Regularization (SNR):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.snr_var = tk.DoubleVar(value=3.0)
        self.snr_scale = tk.Scale(source_frame, from_=1.0, to=10.0, resolution=0.5,
                                  orient=tk.HORIZONTAL, variable=self.snr_var, length=150)
        self.snr_scale.grid(row=2, column=1, padx=10, pady=5)
        
        # Output options
        output_frame = tk.LabelFrame(self.advanced_frame, text="Output Options", 
                                     font=("Arial", 10, "bold"))
        output_frame.pack(pady=10, padx=20, fill="x")
        
        self.save_stc_var = tk.BooleanVar(value=False)
        tk.Checkbutton(output_frame, text="Save source estimates to file",
                       variable=self.save_stc_var).pack(anchor="w", padx=10, pady=2)
        
        self.export_figures_var = tk.BooleanVar(value=False)
        tk.Checkbutton(output_frame, text="Export figures as images",
                       variable=self.export_figures_var).pack(anchor="w", padx=10, pady=2)
        
    def create_results_tab(self):
        """Create results visualization tab"""
        # Results text
        self.results_text = tk.Text(self.results_frame, height=10, wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)
        
    def load_file(self):
        """Load EEG file"""
        filepath = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[("EEG Files", "*.edf *.bdf *.fif *.set *.vhdr"), 
                       ("All files", "*.*")]
        )
        if not filepath:
            return
        
        try:
            self.update_status("Loading EEG file...")
            self.progress['value'] = 10
            
            # Load data
            self.raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            self.raw.pick_types(eeg=True, exclude=[])
            
            # Get file info
            n_channels = len(self.raw.ch_names)
            duration = self.raw.times[-1]
            sfreq = self.raw.info['sfreq']
            self.filename = os.path.basename(filepath)
            
            info_text = (f"Loaded: {self.filename}\n"
                        f"Channels: {n_channels} | Duration: {duration:.1f}s | "
                        f"Sampling rate: {sfreq:.0f} Hz")
            self.file_info_label.config(text=info_text)
            
            # Update time window spinboxes
            self.time_end_var.set(min(10.0, duration))
            self.time_start_spin.config(to=duration)
            self.time_end_spin.config(to=duration)
            
            self.update_status("File loaded successfully. Ready to process.")
            self.progress['value'] = 0
            self.process_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self.progress['value'] = 0
            messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
            self.update_status("Failed to load file.")
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()
        
    def update_progress(self, value, message=""):
        """Update progress bar"""
        self.progress['value'] = value
        if message:
            self.update_status(message)
        self.root.update()
        
    def stop_processing(self):
        """Stop processing"""
        self.stop_requested = True
        self.stop_button.config(state=tk.DISABLED)
        
    def run_reconstruction(self):
        """Run source reconstruction in a separate thread"""
        self.stop_requested = False
        self.stop_button.config(state=tk.NORMAL)
        self.process_button.config(state=tk.DISABLED)
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.notebook.select(self.results_frame)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_data)
        self.processing_thread.start()
        
    def _process_data(self):
        """Main processing function (runs in separate thread)"""
        try:
            total_steps = 10
            current_step = 0
            
            # Check for fsaverage
            self.update_progress(10, "Step 1/10: Checking brain template...")
            subjects_dir = self._check_fsaverage()
            if self.stop_requested:
                return
            
            # Preprocessing
            self.update_progress(20, "Step 2/10: Preprocessing EEG data...")
            raw_processed = self.raw.copy()
            
            # Clean channel names first
            cleaned_names = {name: name.strip().replace('.', '').upper() 
                           for name in raw_processed.ch_names}
            raw_processed.rename_channels(cleaned_names, verbose=False)
            
            # Remove bad channels if requested
            if self.remove_bad_channels_var.get():
                bad_channels = self.preprocessing.detect_bad_channels(raw_processed)
                if bad_channels:
                    # Only mark as bad if they're not frontal channels that are commonly noisy
                    # FP1 and FP2 are often noisy but important for source reconstruction
                    bad_to_remove = []
                    for ch in bad_channels:
                        if ch.upper() not in ['FP1', 'FP2', 'FPZ']:
                            bad_to_remove.append(ch)
                        else:
                            self.log_result(f"Keeping {ch} despite high variance (frontal channel)")
                    
                    if bad_to_remove:
                        self.log_result(f"Detected bad channels: {', '.join(bad_to_remove)}")
                        raw_processed.info['bads'].extend(bad_to_remove)
                        # Don't interpolate yet - do it after montage is set
                    else:
                        self.log_result(f"Detected noisy frontal channels but keeping them: {', '.join(bad_channels)}")
            
            if self.stop_requested:
                return
                
            # Apply montage
            self.update_progress(30, "Step 3/10: Setting channel positions...")
            
            # First, let's see what channels we have
            self.log_result(f"Available channels: {', '.join(raw_processed.ch_names[:10])}{'...' if len(raw_processed.ch_names) > 10 else ''}")
            
            # Try to set montage
            montage_set = False
            montages_to_try = ['standard_1005', 'standard_1020', 'biosemi64', 'biosemi32']
            
            for montage_name in montages_to_try:
                try:
                    montage = mne.channels.make_standard_montage(montage_name)
                    
                    # Find channels that match the montage
                    montage_ch_names_upper = [ch.upper() for ch in montage.ch_names]
                    our_ch_names_upper = [ch.upper() for ch in raw_processed.ch_names]
                    
                    # Find matching channels
                    matching_channels = []
                    for our_ch in raw_processed.ch_names:
                        if our_ch.upper() in montage_ch_names_upper:
                            matching_channels.append(our_ch)
                    
                    if len(matching_channels) >= 10:  # Need reasonable number of channels
                        self.log_result(f"Found {len(matching_channels)} matching channels with {montage_name}")
                        
                        # Keep only matching channels
                        raw_processed.pick_channels(matching_channels, ordered=False)
                        
                        # Now set the montage
                        raw_processed.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
                        
                        # Interpolate bad channels now that we have positions
                        if len(raw_processed.info['bads']) > 0:
                            self.log_result(f"Interpolating bad channels: {', '.join(raw_processed.info['bads'])}")
                            raw_processed.interpolate_bads(reset_bads=True, verbose=False)
                        
                        montage_set = True
                        self.log_result(f"Successfully applied {montage_name} montage")
                        break
                        
                except Exception as e:
                    self.log_result(f"Failed to apply {montage_name}: {str(e)}")
                    continue
            
            if not montage_set:
                raise ValueError("Could not apply any standard montage. Please check your channel names.")
            
            # Verify we have valid positions by checking the data
            n_channels = len(raw_processed.ch_names)
            if n_channels < 10:
                raise ValueError(f"Only {n_channels} channels remaining. Need at least 10 for source reconstruction.")
            
            self.log_result(f"Proceeding with {n_channels} channels")
            
            # Artifact removal
            self.update_progress(40, "Step 4/10: Removing artifacts...")
            raw_processed = self.preprocessing.remove_artifacts(
                raw_processed, self.artifact_removal_var.get())
            
            # Set reference
            raw_processed.set_eeg_reference('average', projection=True, verbose=False)
            
            if self.stop_requested:
                return
                
            # Filter for selected frequency band
            self.update_progress(50, "Step 5/10: Filtering frequency band...")
            freq_band = self.get_frequency_band()
            if freq_band != 'broadband':
                low_freq, high_freq = self.freq_bands[freq_band]
                
                # Check Nyquist frequency
                nyquist = raw_processed.info['sfreq'] / 2.0
                if high_freq > nyquist:
                    self.log_result(f"WARNING: Cannot filter up to {high_freq} Hz with sampling rate {raw_processed.info['sfreq']} Hz")
                    high_freq = nyquist - 1  # Leave 1 Hz margin
                    self.log_result(f"Adjusted high frequency to {high_freq:.1f} Hz")
                
                try:
                    raw_processed.filter(low_freq, high_freq, fir_design='firwin', verbose=False)
                    self.log_result(f"Filtered to {freq_band} band: {low_freq}-{high_freq:.1f} Hz")
                except ValueError as e:
                    # If filtering fails, try with adjusted parameters
                    self.log_result(f"Filter failed: {str(e)}")
                    self.log_result("Trying alternative filter parameters...")
                    raw_processed.filter(low_freq, high_freq, method='iir', verbose=False)
                    self.log_result(f"Applied IIR filter for {freq_band} band")
            
            # Resample based on frequency band needs
            # For gamma band (up to 50 Hz), we need at least 100 Hz sampling
            # Add some headroom for filtering
            freq_band = self.get_frequency_band()
            if freq_band == 'gamma':
                min_sfreq = 150  # 3x the highest frequency for good filtering
            elif freq_band == 'beta':
                min_sfreq = 100  # 3x 30 Hz
            else:
                min_sfreq = 50   # Lower bands don't need high sampling
            
            if raw_processed.info['sfreq'] > min_sfreq * 2:
                # Only resample if we have way more than needed
                new_sfreq = min(200, max(min_sfreq, 100))
                raw_processed.resample(new_sfreq, verbose=False)
                self.log_result(f"Resampled from {self.raw.info['sfreq']:.1f} Hz to {new_sfreq} Hz")
            else:
                self.log_result(f"Keeping original sampling rate: {raw_processed.info['sfreq']:.1f} Hz")
            
            # Crop to selected time window
            tmin = self.time_start_var.get()
            tmax = self.time_end_var.get()
            raw_processed = raw_processed.copy().crop(tmin=tmin, tmax=tmax)
            self.log_result(f"Analyzing time window: {tmin}-{tmax} seconds")
            
            if self.stop_requested:
                return
                
            # Create forward solution
            self.update_progress(60, "Step 6/10: Creating forward solution...")
            fwd = self._create_forward_solution(raw_processed, subjects_dir)
            
            if self.stop_requested:
                return
                
            # Compute inverse operator
            self.update_progress(70, "Step 7/10: Computing inverse operator...")
            inverse_operator = self._compute_inverse_operator(raw_processed, fwd)
            
            if self.stop_requested:
                return
                
            # Apply inverse solution
            self.update_progress(80, "Step 8/10: Reconstructing sources...")
            method = self.method_var.get()
            stc = self.reconstructor.reconstruct(raw_processed, inverse_operator, method)
            self.log_result(f"Applied {method} inverse solution")
            
            # Process visualization type
            self.update_progress(90, "Step 9/10: Processing visualization...")
            stc_viz, params = self._process_visualization(stc)
            
            if self.stop_requested:
                return
                
            # Create visualization
            self.update_progress(95, "Step 10/10: Creating brain visualization...")
            
            # Schedule visualization creation on main thread
            self.root.after(0, lambda sv=stc_viz, p=params, sd=subjects_dir: 
                          self._create_visualization(sv, p, sd))
            
            # Save outputs if requested
            if self.save_stc_var.get():
                stc_filename = f"{self.filename.split('.')[0]}_stc"
                stc.save(stc_filename, verbose=False)
                self.log_result(f"Saved source estimates to: {stc_filename}")
            
            self.update_progress(100, "Processing complete!")
            self.log_result("\n✓ Source reconstruction completed successfully!")
            
        except Exception as e:
            self.update_progress(0, f"Error: {str(e)}")
            self.log_result(f"\n✗ Error during processing: {str(e)}")
            # Schedule error dialog on main thread
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
            
        finally:
            self.process_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
    
    def _check_fsaverage(self):
        """Check and download fsaverage if needed"""
        # Similar to original but with better error handling
        try:
            subjects_dir = mne.get_subjects_dir()
        except:
            subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data', 'MNE-fsaverage-data')
        
        fsaverage_path = os.path.join(subjects_dir, 'fsaverage')
        if not os.path.exists(fsaverage_path):
            self.log_result("Downloading fsaverage brain template...")
            try:
                mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
            except:
                os.makedirs(fsaverage_path, exist_ok=True)
        
        return subjects_dir
    
    def _create_forward_solution(self, raw, subjects_dir):
        """Create forward solution"""
        # Create BEM model
        model = mne.make_bem_model(
            subject='fsaverage', 
            ico=4,
            conductivity=(0.3, 0.006, 0.3),  # brain, skull, scalp
            subjects_dir=subjects_dir, 
            verbose=False
        )
        bem_sol = mne.make_bem_solution(model, verbose=False)
        
        # Setup source space
        spacing = self.spacing_var.get()
        src = mne.setup_source_space(
            'fsaverage', 
            spacing=spacing,
            add_dist=False,
            subjects_dir=subjects_dir, 
            verbose=False
        )
        self.log_result(f"Created source space with {spacing} spacing")
        
        # Make forward solution
        fwd = mne.make_forward_solution(
            raw.info, 
            trans='fsaverage',
            src=src, 
            bem=bem_sol,
            meg=False, 
            eeg=True,
            mindist=5.0, 
            verbose=False
        )
        
        return fwd
    
    def _compute_inverse_operator(self, raw, fwd):
        """Compute inverse operator"""
        # Compute noise covariance
        noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
        
        # Convert SNR to lambda2
        snr = self.snr_var.get()
        lambda2 = 1.0 / snr ** 2
        
        # Make inverse operator
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            raw.info, fwd, noise_cov,
            loose=0.2, 
            depth=0.8,
            verbose=False
        )
        
        self.log_result(f"Created inverse operator with SNR={snr}")
        return inverse_operator
    
    def _process_visualization(self, stc):
        """Process visualization based on selected type"""
        viz_type = self.viz_var.get()
        
        if viz_type == "phase":
            # Calculate instantaneous phase
            phase_data = np.angle(hilbert(stc.data, axis=1))
            stc_viz = mne.SourceEstimate(
                phase_data, vertices=stc.vertices,
                tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage'
            )
            params = {
                'colormap': 'twilight_shifted',
                'clim': dict(kind='value', lims=[-np.pi, 0, np.pi]),
                'title': f"Phase Patterns - {self.freq_var.get().title()} Band"
            }
            
        elif viz_type == "power":
            # Calculate power
            power_data = stc.data ** 2
            stc_viz = mne.SourceEstimate(
                power_data, vertices=stc.vertices,
                tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage'
            )
            params = {
                'colormap': 'hot',
                'clim': dict(kind='percent', lims=[70, 85, 99]),
                'title': f"Power Distribution - {self.freq_var.get().title()} Band"
            }
            
        elif viz_type == "stats":
            # Statistical thresholding (simple z-score)
            mean_data = np.mean(stc.data, axis=1, keepdims=True)
            std_data = np.std(stc.data, axis=1, keepdims=True)
            z_scores = (stc.data - mean_data) / (std_data + 1e-10)
            
            # Threshold at |z| > 2
            z_scores[np.abs(z_scores) < 2] = 0
            
            stc_viz = mne.SourceEstimate(
                z_scores, vertices=stc.vertices,
                tmin=stc.tmin, tstep=stc.tstep, subject='fsaverage'
            )
            params = {
                'colormap': 'RdBu_r',
                'clim': dict(kind='value', lims=[-4, 0, 4]),
                'title': f"Statistical Map (z-scores) - {self.freq_var.get().title()} Band"
            }
            
        else:  # raw
            stc_viz = stc
            params = {
                'colormap': 'RdBu_r',
                'clim': dict(kind='percent', lims=[5, 50, 95]),
                'title': f"Source Amplitude - {self.freq_var.get().title()} Band"
            }
        
        return stc_viz, params
    
    def _create_visualization(self, stc, params, subjects_dir):
        """Create brain visualization (must be called on main thread)"""
        try:
            # Create brain plot using the backend set at startup
            brain = stc.plot(
                subjects_dir=subjects_dir,
                subject='fsaverage',
                surface='pial',
                hemi='both',
                colormap=params['colormap'],
                clim=params['clim'],
                time_label=f"{params['title']} (t=%0.2f s)",
                size=(1200, 800),
                smoothing_steps=5,
                background='white',
                verbose=False
            )
            
            # Add title and info
            brain.add_text(0.1, 0.9, params['title'], 'title', font_size=16)
            info_text = f"{self.method_var.get()} | {len(stc.vertices[0]+stc.vertices[1])} sources"
            brain.add_text(0.1, 0.05, info_text, 'info', font_size=10)
            
            backend = mne.viz.get_3d_backend()
            self.log_result(f"Created 3D visualization with {backend}")
            
            # Keep reference to prevent garbage collection
            self.brain_figures.append(brain)
            
            # Export if requested
            if self.export_figures_var.get():
                # Schedule screenshot after a delay to ensure rendering
                self.root.after(1500, lambda: self._export_brain_figure(brain, params))
            
            # Show instructions
            self.log_result("\n📌 3D Brain Visualization Controls:")
            self.log_result("• Left click + drag: Rotate brain")
            self.log_result("• Right click + drag: Zoom in/out")
            self.log_result("• Middle click + drag: Pan view")
            self.log_result("• Spacebar: Start/stop time animation")
            self.log_result("• Use time slider to explore different time points")
            self.log_result("\n✅ Brain window should now stay open!")
                
        except Exception as e:
            self.log_result(f"3D visualization error: {str(e)}")
            self.log_result("Creating 2D visualization instead...")
            self._create_2d_visualization(stc, params)
    
    def _export_brain_figure(self, brain, params):
        """Export brain figure (called after delay)"""
        try:
            screenshot = brain.screenshot()
            filename = f"{self.filename.split('.')[0]}_{self.viz_var.get()}.png"
            plt.imsave(filename, screenshot)
            self.log_result(f"Saved figure: {filename}")
        except Exception as e:
            self.log_result(f"Failed to save screenshot: {str(e)}")
    
    def _create_2d_visualization(self, stc, params):
        """Create 2D fallback visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(params['title'], fontsize=14)
        
        # Time series plot
        ax = axes[0, 0]
        times_to_plot = np.linspace(stc.times[0], stc.times[-1], 100)
        mean_activity = stc.data.mean(axis=0)
        std_activity = stc.data.std(axis=0)
        
        ax.plot(stc.times, mean_activity, 'b-', label='Mean')
        ax.fill_between(stc.times, 
                       mean_activity - std_activity,
                       mean_activity + std_activity,
                       alpha=0.3, color='blue', label='±1 STD')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Source Activity')
        ax.set_title('Average Source Activity Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Histogram at different time points
        time_points = [0.25, 0.5, 0.75]
        for i, frac in enumerate(time_points):
            ax = axes[i//2, i%2+1]
            time_idx = int(frac * len(stc.times))
            data_at_time = stc.data[:, time_idx]
            
            ax.hist(data_at_time, bins=50, alpha=0.7, color='green', edgecolor='black')
            ax.set_xlabel(f'{self.viz_var.get().title()} Value')
            ax.set_ylabel('Number of Sources')
            ax.set_title(f'Distribution at t={stc.times[time_idx]:.2f}s')
            ax.grid(True, alpha=0.3)
        
        # Power spectrum of sources
        ax = axes[1, 0]
        # Simple frequency content visualization
        fft_data = np.abs(np.fft.rfft(stc.data.mean(axis=0)))
        freqs = np.fft.rfftfreq(len(stc.times), stc.tstep)
        
        ax.semilogy(freqs[:50], fft_data[:50], 'r-')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title('Frequency Content of Source Activity')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 50)
        
        plt.tight_layout()
        
        # Export if requested
        if self.export_figures_var.get():
            filename = f"{self.filename.split('.')[0]}_{self.viz_var.get()}_2d.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            self.log_result(f"Saved 2D figure: {filename}")
        
        plt.show()
    
    def get_frequency_band(self):
        """Get selected frequency band"""
        return self.freq_var.get()
    
    @property
    def freq_bands(self):
        """Frequency band definitions"""
        return {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "beta": (12, 30),
            "gamma": (30, 50),
            "broadband": (0.5, 50)
        }
    
    def close_brain_views(self):
        """Close all open brain visualization windows"""
        for brain in self.brain_figures:
            try:
                brain.close()
            except:
                pass
        self.brain_figures.clear()
        self.log_result("Closed all 3D brain views")
    
    def log_result(self, message):
        """Log message to results tab"""
        self.results_text.insert(tk.END, f"{message}\n")
        self.results_text.see(tk.END)
        self.root.update()

def check_dependencies():
    """Check and report on available dependencies"""
    print("EEG Brain Source Reconstruction Tool - Enhanced Version")
    print("=" * 50)
    print("Checking dependencies...")
    
    dependencies = {
        'MNE-Python': 'mne',
        'NumPy': 'numpy',
        'SciPy': 'scipy',
        'Matplotlib': 'matplotlib',
        'PyVista': 'pyvista',
        'PyVistaQt': 'pyvistaqt',
        'Mayavi': 'mayavi'
    }
    
    available_3d = []
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} is installed")
            if module in ['pyvista', 'pyvistaqt', 'mayavi']:
                available_3d.append(module)
        except ImportError:
            print(f"✗ {name} is not installed")
            if module in ['mne', 'numpy', 'scipy', 'matplotlib']:
                print(f"  REQUIRED: Install with 'pip install {module}'")
    
    print("\n3D Visualization Status:")
    if available_3d:
        print(f"✓ 3D visualization available with: {', '.join(available_3d)}")
    else:
        print("⚠ No 3D backends installed. 2D fallback will be used.")
        print("  For 3D visualization, install: pip install pyvista pyvistaqt")
    
    print("\nStarting application...\n")

if __name__ == "__main__":
    # Check dependencies and provide feedback
    check_dependencies()
    
    # Set up 3D backend if available
    try:
        import pyvistaqt
        mne.viz.set_3d_backend("pyvistaqt")
    except:
        try:
            import pyvista
            mne.viz.set_3d_backend("pyvista")
        except:
            try:
                import mayavi
                mne.viz.set_3d_backend("mayavi")
            except:
                print("Note: Using 2D fallback visualization")
    
    # Create and run application
    root = tk.Tk()
    app = EEGSourceReconstructionApp(root)
    root.mainloop()