# EEG Brain Source & Coordination Explorer with Harmonic Decomposition
#
# This version integrates the "Universal Brain Coordination Model" to visualize
# conductor power, multi-band harmony, phase-slip dynamics, and the new
# "Coordinated Power" (Y*PLV) metric in source space.
# Additionally, it includes Connectome Harmonic analysis to separate
# Structure-Coupled vs Innovation (Decoupled) signals.

import os
import sys
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import mne
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging
import threading
import traceback

# Suppress verbose MNE logging and Qt warnings
logging.basicConfig(level=logging.WARNING)
mne.set_log_level('WARNING')
warnings.filterwarnings('ignore', message='.*QApplication.*')
warnings.filterwarnings('ignore', message='.*QWindowsWindow.*')

# Replace the entire HarmonicEngine class with this corrected version:

class HarmonicEngine:
    """
    Connectome Harmonic Engine that works on reduced source spaces.
    Uses MNE's source space triangulation directly.
    """
    
    def __init__(self, subjects_dir=None):
        if subjects_dir is None:
            self.subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
        else:
            self.subjects_dir = subjects_dir
            
        if not os.path.exists(self.subjects_dir):
            os.makedirs(self.subjects_dir)
            
        self.subject = 'fsaverage'
        self.src = None
        self.harmonic_modes = None
        self.eigenvalues = None
        self.n_modes = 100
        self.spacing = 'ico5'
        self.initialized = False
        
    def initialize(self, spacing='ico5', n_modes=100):
        """
        Initialize harmonic engine on the REDUCED MNE source space (ico / oct).
        Uses vertno + use_tris only. Never touches full fsaverage geometry.
        """
        import mne
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.linalg import eigsh

        self.spacing = spacing
        self.n_modes = n_modes

        print("Checking for fsaverage...")
        mne.datasets.fetch_fsaverage(subjects_dir=self.subjects_dir, verbose=False)

        print(f"Creating source space with {spacing} spacing...")
        self.src = mne.setup_source_space(
            self.subject,
            spacing=spacing,
            subjects_dir=self.subjects_dir,
            add_dist=False,
            verbose=False
        )

        # ---- DEBUG / SANITY CHECK (ONCE) ----
        print("\nSource space summary:")
        total_vertices = 0
        for hemi_idx, surf in enumerate(self.src):
            hemi = "LH" if hemi_idx == 0 else "RH"
            n = len(surf['vertno'])
            total_vertices += n
            print(f"  {hemi}: {n} vertices")

        print(f"Total source-space vertices: {total_vertices}")

        expected_counts = {
            'ico4': 5124,
            'ico5': 20484,
            'oct5': 40968,
            'oct6': 163848,
        }
        if spacing in expected_counts:
            if total_vertices != expected_counts[spacing]:
                print(f"⚠ WARNING: Expected {expected_counts[spacing]}, got {total_vertices}")
            else:
                print(f"✓ Vertex count matches expected for {spacing}")

        # ---- COMPUTE HARMONICS ----
        self.harmonic_modes = []
        self.eigenvalues = []

        for hemi_idx, surf in enumerate(self.src):
            hemi = "Left" if hemi_idx == 0 else "Right"
            print(f"\nProcessing {hemi} hemisphere...")

            vertno = surf['vertno']
            n_vertices = len(vertno)
            print(f"  Using {n_vertices} vertices")

            # --- TRIANGULATION (ALREADY IN REDUCED SPACE) ---
            tris = surf.get('use_tris', None)
            if tris is None or len(tris) == 0:
                raise RuntimeError(
                    f"No triangulation available for {hemi} hemisphere "
                    f"(spacing={spacing})"
                )

            print(f"  Using {len(tris)} triangles")

            # --- BUILD ADJACENCY ---
            row = np.concatenate([
                tris[:, 0], tris[:, 1],
                tris[:, 1], tris[:, 2],
                tris[:, 2], tris[:, 0]
            ])
            col = np.concatenate([
                tris[:, 1], tris[:, 0],
                tris[:, 2], tris[:, 1],
                tris[:, 0], tris[:, 2]
            ])

            data = np.ones(len(row), dtype=np.float32)
            A = sp.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices))
            A = A.tocsr()
            A.data[:] = 1.0
            A.setdiag(0)
            A.eliminate_zeros()
            A = A.maximum(A.T)

            print(f"  Adjacency edges: {A.nnz}")

            # --- GRAPH LAPLACIAN ---
            d = np.array(A.sum(axis=1)).ravel()
            D = sp.diags(d, dtype=np.float32)
            L = (D - A).astype(np.float32)

            # --- EIGENSOLVE ---
            k = min(self.n_modes, n_vertices - 2)
            print(f"  Solving {k} eigenmodes...")

            try:
                L_reg = L + 1e-8 * sp.identity(n_vertices, dtype=np.float32)
                evals, evecs = eigsh(
                    L_reg,
                    k=k,
                    which='SM',
                    tol=1e-4,
                    maxiter=5000
                )
                print("  ✓ Sparse eigensolver OK")
            except Exception as e:
                print(f"  ⚠ Sparse failed, using dense: {e}")
                L_dense = L.toarray()
                evals, evecs = np.linalg.eigh(L_dense)
                evals = evals[:k]
                evecs = evecs[:, :k]

            self.eigenvalues.append(evals)
            self.harmonic_modes.append(evecs)

            print(f"  Stored modes: {evecs.shape}")

        # ---- FINAL CONSISTENCY CHECK ----
        expected = sum(len(s['vertno']) for s in self.src)
        actual = sum(m.shape[0] for m in self.harmonic_modes)

        print("\nMesh check:")
        print(f"  Source space vertices = {expected}")
        print(f"  Harmonic modes vertices = {actual}")

        assert expected == actual, (
            f"Harmonic mesh mismatch: expected {expected}, got {actual}"
        )

        self.initialized = True
        print("✓ Harmonic engine initialized correctly")
        return self.src

        
    def check_dimensions(self, stc):
        """
        Check if STC dimensions match harmonic engine dimensions.
        """
        if not self.initialized:
            return False
            
        n_vertices_total = stc.data.shape[0]
        n_vertices_per_hemi_lh = self.harmonic_modes[0].shape[0]
        n_vertices_per_hemi_rh = self.harmonic_modes[1].shape[0]
        total_expected = n_vertices_per_hemi_lh + n_vertices_per_hemi_rh
        
        print(f"\nDimension check:")
        print(f"  STC total vertices: {n_vertices_total}")
        print(f"  Harmonic left: {n_vertices_per_hemi_lh}")
        print(f"  Harmonic right: {n_vertices_per_hemi_rh}")
        print(f"  Expected total: {total_expected}")
        
        # Also check STC vertex counts
        if hasattr(stc, 'lh_vertno'):
            stc_lh_vertices = len(stc.lh_vertno) if stc.lh_vertno is not None else "N/A"
            stc_rh_vertices = len(stc.rh_vertno) if stc.rh_vertno is not None else "N/A"
            print(f"  STC left vertices: {stc_lh_vertices}")
            print(f"  STC right vertices: {stc_rh_vertices}")
        
        match = n_vertices_total == total_expected
        print(f"  Match: {'✓' if match else '✗'}")
        
        return match
        
    def project_stc_to_harmonics(self, stc):
        """
        Perform Graph Fourier Transform (GFT) on SourceEstimate data.
        """
        if not self.initialized:
            raise ValueError("HarmonicEngine not initialized. Call initialize() first.")
            
        # First check dimensions
        self.check_dimensions(stc)
            
        # Get total number of vertices in STC
        n_vertices_total = stc.data.shape[0]
        
        # Get number of vertices per hemisphere from harmonic modes
        n_vertices_per_hemi_lh = self.harmonic_modes[0].shape[0]
        n_vertices_per_hemi_rh = self.harmonic_modes[1].shape[0]
        total_expected = n_vertices_per_hemi_lh + n_vertices_per_hemi_rh
        
        # Check if dimensions match
        if n_vertices_total != total_expected:
            # Try to get more detailed information
            if hasattr(stc, 'lh_data') and hasattr(stc, 'rh_data'):
                stc_lh_vertices = stc.lh_data.shape[0] if stc.lh_data is not None else 0
                stc_rh_vertices = stc.rh_data.shape[0] if stc.rh_data is not None else 0
                stc_total = stc_lh_vertices + stc_rh_vertices
                print(f"STC detailed: LH={stc_lh_vertices}, RH={stc_rh_vertices}, Total={stc_total}")
            
            raise ValueError(
                f"Dimension mismatch: STC has {n_vertices_total} vertices, "
                f"but harmonic modes expect {total_expected} vertices.\n"
                f"STC left vertices: {stc.data[:n_vertices_per_hemi_lh].shape[0] if n_vertices_total > n_vertices_per_hemi_lh else 'N/A'}\n"
                f"STC right vertices: {stc.data[n_vertices_per_hemi_lh:].shape[0] if n_vertices_total > n_vertices_per_hemi_lh else 'N/A'}\n"
                f"Harmonic left vertices: {n_vertices_per_hemi_lh}\n"
                f"Harmonic right vertices: {n_vertices_per_hemi_rh}\n"
                f"Make sure source space spacing ({self.spacing}) is consistent."
            )
        
        # Extract hemisphere data
        lh_data = stc.data[:n_vertices_per_hemi_lh, :]
        rh_data = stc.data[n_vertices_per_hemi_lh:, :]
        
        print(f"Projecting to harmonics...")
        print(f"  LH data shape: {lh_data.shape}")
        print(f"  RH data shape: {rh_data.shape}")
        print(f"  LH modes shape: {self.harmonic_modes[0].shape}")
        print(f"  RH modes shape: {self.harmonic_modes[1].shape}")
        
        # GFT: x_hat = U^T * x
        lh_harmonics = np.dot(self.harmonic_modes[0].T, lh_data)
        rh_harmonics = np.dot(self.harmonic_modes[1].T, rh_data)
        
        print(f"  LH harmonics shape: {lh_harmonics.shape}")
        print(f"  RH harmonics shape: {rh_harmonics.shape}")
        
        return lh_harmonics, rh_harmonics
        
    def analyze_structure_function_coupling(self, stc, k_cutoff=10):
        """
        Separate EEG signal into Structure-Coupled and Innovation (Decoupled) components.
        """
        if not self.initialized:
            raise ValueError("HarmonicEngine not initialized. Call initialize() first.")
            
        print(f"\nAnalyzing structure-function coupling...")
        print(f"  Using k_cutoff = {k_cutoff}")
            
        # Project to harmonic space
        lh_h, rh_h = self.project_stc_to_harmonics(stc)
        
        # Get number of vertices per hemisphere
        n_vertices_per_hemi_lh = self.harmonic_modes[0].shape[0]
        n_vertices_per_hemi_rh = self.harmonic_modes[1].shape[0]
        
        # Extract hemisphere data
        lh_data = stc.data[:n_vertices_per_hemi_lh, :]
        rh_data = stc.data[n_vertices_per_hemi_lh:, :]
        
        print(f"  LH original shape: {lh_data.shape}")
        print(f"  RH original shape: {rh_data.shape}")
        
        # Reconstruct coupled signal using only the first 'k' modes
        print(f"  Reconstructing coupled signal...")
        lh_coupled = np.dot(self.harmonic_modes[0][:, :k_cutoff], lh_h[:k_cutoff, :])
        rh_coupled = np.dot(self.harmonic_modes[1][:, :k_cutoff], rh_h[:k_cutoff, :])
        
        # The 'Innovation' signal is the residual (Total - Coupled)
        print(f"  Computing innovation signal...")
        lh_innovation = lh_data - lh_coupled
        rh_innovation = rh_data - rh_coupled
        
        print(f"  LH coupled shape: {lh_coupled.shape}")
        print(f"  RH coupled shape: {rh_coupled.shape}")
        print(f"  LH innovation shape: {lh_innovation.shape}")
        print(f"  RH innovation shape: {rh_innovation.shape}")
        
        return (lh_coupled, rh_coupled), (lh_innovation, rh_innovation)
        
    # Rest of the methods remain the same...
        
    def get_harmonic_mode_stc(self, mode_idx=0, time_points=1):
        """
        Create a SourceEstimate object for visualizing a specific harmonic mode.
        
        Parameters:
        -----------
        mode_idx : int, optional
            Index of harmonic mode to visualize (default: 0)
        time_points : int, optional
            Number of time points to create (default: 1)
            
        Returns:
        --------
        stc_mode : mne.SourceEstimate
            SourceEstimate containing the harmonic mode pattern
        """
        if not self.initialized:
            raise ValueError("HarmonicEngine not initialized. Call initialize() first.")
            
        # Ensure mode_idx is valid
        max_modes = min(self.harmonic_modes[0].shape[1], self.harmonic_modes[1].shape[1])
        if mode_idx >= max_modes:
            mode_idx = max_modes - 1
            
        # Get mode data for each hemisphere
        lh_mode = self.harmonic_modes[0][:, mode_idx:mode_idx+1]
        rh_mode = self.harmonic_modes[1][:, mode_idx:mode_idx+1]
        
        # Repeat for desired number of time points
        if time_points > 1:
            lh_mode = np.repeat(lh_mode, time_points, axis=1)
            rh_mode = np.repeat(rh_mode, time_points, axis=1)
        
        # Combine hemispheres
        data = np.vstack([lh_mode, rh_mode])
        
        # Create vertices list
        vertices = [self.src[0]['vertno'], self.src[1]['vertno']]
        
        # Create SourceEstimate
        stc_mode = mne.SourceEstimate(
            data, 
            vertices=vertices, 
            tmin=0, 
            tstep=1, 
            subject=self.subject
        )
        
        return stc_mode
        
    def plot_eigenspectrum(self, ax=None):
        """
        Plot the eigenvalue spectrum of the graph Laplacian.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes with the plot
        """
        if not self.initialized:
            raise ValueError("HarmonicEngine not initialized. Call initialize() first.")
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        # Plot eigenvalues for both hemispheres
        colors = ['blue', 'red']
        labels = ['Left Hemisphere', 'Right Hemisphere']
        for hemi_idx, evals in enumerate(self.eigenvalues):
            ax.plot(range(1, len(evals)+1), evals, 'o-', color=colors[hemi_idx], 
                label=labels[hemi_idx], alpha=0.7, markersize=4)
            
        ax.set_xlabel('Mode Index', fontsize=12)
        ax.set_ylabel(r'Eigenvalue ($\lambda$)', fontsize=12)  # FIXED: Added 'r' prefix
        ax.set_title(f'Graph Laplacian Eigenspectrum ({self.spacing})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax

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
        # Use median and median absolute deviation for robustness against outliers
        median_var = np.median(channel_vars)
        mad = np.median(np.abs(channel_vars - median_var))
        if mad == 0:
            return []
        z_scores = 0.6745 * np.abs(channel_vars - median_var) / mad
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
            except Exception as e:
                print(f"ICA failed, falling back to basic filtering: {e}")
                # Fallback to basic if ICA fails
                PreprocessingPipeline.remove_artifacts(raw, 'basic')
        
        return raw

class EEGSourceReconstructionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Brain Source & Harmonic Explorer")
        self.root.geometry("900x1000")
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Initialize components
        self.reconstructor = SourceReconstructor()
        self.preprocessing = PreprocessingPipeline()
        self.processing_thread = None
        self.brain_figures = []  # Keep track of brain figures
        self.harmonic_engine = None  # Will be initialized when needed
        self.harmonic_modes_computed = False
        
        # Initialize subjects directory
        self.subjects_dir = os.path.join(os.path.expanduser('~'), 'mne_data')
        
        # File variables
        self.raw = None
        self.filename = ""
        
        # Stop flag
        self.stop_requested = False
        
        # Create UI
        self.create_ui()
        
    def create_ui(self):
        """Create the user interface"""
        # Title
        title_label = tk.Label(self.root, text="EEG Brain Source & Harmonic Explorer", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Analysis type frame
        analysis_type_frame = tk.Frame(self.root)
        analysis_type_frame.pack(pady=5)
        self.analysis_mode = tk.StringVar(value="harmonic")  # Default to harmonic mode
        tk.Radiobutton(analysis_type_frame, text="Standard Analysis", 
                      variable=self.analysis_mode, value="standard", 
                      command=self.switch_tabs).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(analysis_type_frame, text="Coordination Model", 
                      variable=self.analysis_mode, value="coordination", 
                      command=self.switch_tabs).pack(side=tk.LEFT, padx=10)
        tk.Radiobutton(analysis_type_frame, text="Harmonic Coupling", 
                      variable=self.analysis_mode, value="harmonic", 
                      command=self.switch_tabs).pack(side=tk.LEFT, padx=10)

        # Standard Analysis tab
        self.standard_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.standard_frame, text="Standard Settings")
        self.create_standard_tab()

        # Coordination Model tab
        self.coordination_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.coordination_frame, text="Coordination Model Settings")
        self.create_coordination_tab()
        
        # Harmonic Analysis tab
        self.harmonic_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.harmonic_frame, text="Harmonic Settings")
        self.create_harmonic_tab()
        
        # Advanced tab (common to all)
        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced Settings")
        self.create_advanced_tab()
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Log & Results")
        self.create_results_tab()
        
        # Eigenspectrum tab (for harmonic analysis)
        self.spectrum_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.spectrum_frame, text="Eigenspectrum")
        self.create_spectrum_tab()
        
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
        
        self.compute_modes_button = tk.Button(button_frame, text="Compute Harmonic Modes", 
                                             command=self.compute_harmonic_modes,
                                             bg="#9C27B0", fg="white", font=("Arial", 10))
        self.compute_modes_button.pack(side=tk.LEFT, padx=5)
        
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
                                     wraplength=850)
        self.status_label.pack(pady=5)
        
        # File info
        self.file_info_label = tk.Label(self.root, text="", fg="blue", wraplength=850)
        self.file_info_label.pack(pady=5)

        self.switch_tabs()

    def switch_tabs(self):
        """Show/hide tabs based on analysis mode"""
        # Always show Advanced, Results, and Spectrum tabs
        for tab in [self.advanced_frame, self.results_frame, self.spectrum_frame]:
            self.notebook.add(tab)
            
        if self.analysis_mode.get() == "standard":
            self.notebook.add(self.standard_frame)
            self.notebook.hide(self.coordination_frame)
            self.notebook.hide(self.harmonic_frame)
        elif self.analysis_mode.get() == "coordination":
            self.notebook.add(self.coordination_frame)
            self.notebook.hide(self.standard_frame)
            self.notebook.hide(self.harmonic_frame)
        else:  # harmonic
            self.notebook.add(self.harmonic_frame)
            self.notebook.hide(self.standard_frame)
            self.notebook.hide(self.coordination_frame)

    def create_standard_tab(self):
        """Create basic settings tab"""
        # Frequency band selection
        freq_frame = tk.LabelFrame(self.standard_frame, text="Frequency Band", 
                                   font=("Arial", 10, "bold"))
        freq_frame.pack(pady=10, padx=20, fill="x")
        
        self.freq_var = tk.StringVar(value="alpha")
        freq_options = [
            ("Delta (0.5-4 Hz)", "delta"), ("Theta (4-8 Hz)", "theta"), 
            ("Alpha (8-12 Hz)", "alpha"), ("Beta (12-30 Hz)", "beta"),
            ("Gamma (30-50 Hz)", "gamma"), ("Broadband (0.5-50 Hz)", "broadband")
        ]
        
        for i, (text, value) in enumerate(freq_options):
            tk.Radiobutton(freq_frame, text=text, variable=self.freq_var, 
                           value=value, font=("Arial", 9)).grid(row=i//2, column=i%2, 
                                                               sticky="w", padx=10, pady=2)
        
        # Visualization type
        viz_frame = tk.LabelFrame(self.standard_frame, text="Visualization Type", 
                                  font=("Arial", 10, "bold"))
        viz_frame.pack(pady=10, padx=20, fill="x")
        
        self.viz_var = tk.StringVar(value="power")
        viz_options = [
            ("Power Distribution", "power"), ("Phase Patterns", "phase"),
            ("Raw Amplitude", "raw"), ("Statistical Map (z-score)", "stats")
        ]
        
        for i, (text, value) in enumerate(viz_options):
            tk.Radiobutton(viz_frame, text=text, variable=self.viz_var, 
                           value=value, font=("Arial", 9)).pack(anchor="w", padx=10)
    
    def create_coordination_tab(self):
        """Create coordination model settings tab"""
        # Conductor selection
        conductor_frame = tk.LabelFrame(self.coordination_frame, text="Conductor Frequency",
                                        font=("Arial", 10, "bold"))
        conductor_frame.pack(pady=10, padx=20, fill="x")
        
        self.conductor_var = tk.StringVar(value="alpha")
        conductor_options = [
            ("Alpha (8-12 Hz)", "alpha"), ("Gamma (30-50 Hz)", "gamma"),
            ("Beta (12-30 Hz)", "beta"), ("Theta (4-8 Hz)", "theta")
        ]
        for text, value in conductor_options:
            tk.Radiobutton(conductor_frame, text=text, variable=self.conductor_var,
                           value=value).pack(anchor="w", padx=10)

        # Orchestra (Moiré) selection
        orchestra_frame = tk.LabelFrame(self.coordination_frame, text="Orchestra (Moiré Composite) Frequencies",
                                        font=("Arial", 10, "bold"))
        orchestra_frame.pack(pady=10, padx=20, fill="x")
        self.orchestra_vars = {
            "delta": tk.BooleanVar(value=True), "theta": tk.BooleanVar(value=True),
            "alpha": tk.BooleanVar(value=True), "beta": tk.BooleanVar(value=True),
            "gamma": tk.BooleanVar(value=True)
        }
        for i, band in enumerate(self.orchestra_vars):
            tk.Checkbutton(orchestra_frame, text=f"{band.title()}",
                           variable=self.orchestra_vars[band]).grid(row=0, column=i, padx=5)

        # Visualization Metric
        metric_frame = tk.LabelFrame(self.coordination_frame, text="Visualization Metric",
                                     font=("Arial", 10, "bold"))
        metric_frame.pack(pady=10, padx=20, fill="x")

        metric_options = [
            ("X-Axis: Conductor Power", "Conductor Power"),
            ("Y-Axis: Moiré Harmony", "Moiré Harmony"),
            ("Z-Axis: Phase-Slip Rate", "Phase-Slip Rate"),
            ("Coordinated Power (Y*PLV)", "Coordinated Power (Y*PLV)")
        ]
        self.coord_viz_var = tk.StringVar()
        for text, value in metric_options:
            tk.Radiobutton(metric_frame, text=text, variable=self.coord_viz_var,
                           value=value).pack(anchor="w", padx=10)
        self.coord_viz_var.set("Coordinated Power (Y*PLV)")
        
    def create_harmonic_tab(self):
        """Create harmonic analysis settings tab"""
        # Harmonic analysis type
        harmonic_type_frame = tk.LabelFrame(self.harmonic_frame, text="Harmonic Analysis Type",
                                           font=("Arial", 10, "bold"))
        harmonic_type_frame.pack(pady=10, padx=20, fill="x")
        
        self.harmonic_type_var = tk.StringVar(value="innovation")
        harmonic_options = [
            ("Structure-Coupled Signal", "coupled"),
            ("Innovation (Decoupled) Signal", "innovation"),
            ("Specific Harmonic Mode", "mode"),
            ("Harmonic Power Spectrum", "spectrum")
        ]
        
        for text, value in harmonic_options:
            tk.Radiobutton(harmonic_type_frame, text=text, variable=self.harmonic_type_var,
                          value=value, font=("Arial", 9)).pack(anchor="w", padx=10, pady=2)
        
        # Mode selection (for specific mode visualization)
        mode_frame = tk.LabelFrame(self.harmonic_frame, text="Harmonic Mode Selection",
                                  font=("Arial", 10, "bold"))
        mode_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(mode_frame, text="Mode Index (0-99):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.mode_idx_var = tk.IntVar(value=0)
        self.mode_idx_spin = tk.Spinbox(mode_frame, from_=0, to=99, increment=1,
                                       textvariable=self.mode_idx_var, width=10)
        self.mode_idx_spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Coupling cutoff
        cutoff_frame = tk.LabelFrame(self.harmonic_frame, text="Structure-Coupling Cutoff",
                                    font=("Arial", 10, "bold"))
        cutoff_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(cutoff_frame, text="Number of low-frequency modes for coupling:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.k_cutoff_var = tk.IntVar(value=10)
        self.k_cutoff_spin = tk.Spinbox(cutoff_frame, from_=1, to=50, increment=1,
                                       textvariable=self.k_cutoff_var, width=10)
        self.k_cutoff_spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Harmonic visualization settings
        viz_frame = tk.LabelFrame(self.harmonic_frame, text="Visualization Settings",
                                 font=("Arial", 10, "bold"))
        viz_frame.pack(pady=10, padx=20, fill="x")
        
        self.harmonic_colormap_var = tk.StringVar(value="RdBu_r")
        tk.Label(viz_frame, text="Colormap:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        colormap_menu = ttk.Combobox(viz_frame, textvariable=self.harmonic_colormap_var,
                                    values=["RdBu_r", "viridis", "plasma", "hot", "coolwarm", "twilight"],
                                    state="readonly", width=15)
        colormap_menu.grid(row=0, column=1, padx=5, pady=5, sticky="w")

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
        
        # Harmonic engine settings
        harmonic_frame = tk.LabelFrame(self.advanced_frame, text="Harmonic Engine Settings",
                                      font=("Arial", 10, "bold"))
        harmonic_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(harmonic_frame, text="Number of modes to compute:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.n_modes_var = tk.IntVar(value=100)
        n_modes_spin = tk.Spinbox(harmonic_frame, from_=10, to=200, increment=10,
                                 textvariable=self.n_modes_var, width=10)
        n_modes_spin.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Time window
        time_frame = tk.LabelFrame(self.advanced_frame, text="Time Window (seconds)",
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

    def create_results_tab(self):
        """Create results visualization tab"""
        # Create a frame with scrollbars
        results_container = tk.Frame(self.results_frame)
        results_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Text widget for logs
        self.results_text = tk.Text(results_container, wrap=tk.WORD, bg="#f0f0f0")
        self.results_text.pack(side=tk.LEFT, fill="both", expand=True)
        
        # Vertical scrollbar
        v_scrollbar = ttk.Scrollbar(results_container, orient="vertical", command=self.results_text.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=v_scrollbar.set)
        
        # Horizontal scrollbar
        h_scrollbar = ttk.Scrollbar(self.results_frame, orient="horizontal", command=self.results_text.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_text.config(xscrollcommand=h_scrollbar.set, wrap=tk.NONE)

    def create_spectrum_tab(self):
        """Create eigenspectrum visualization tab"""
        self.spectrum_canvas = None
        self.spectrum_figure = None
        
        # Frame for matplotlib figure
        spectrum_container = tk.Frame(self.spectrum_frame)
        spectrum_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Label placeholder
        self.spectrum_label = tk.Label(spectrum_container, text="Eigenspectrum will appear here after computing harmonic modes.",
                                      font=("Arial", 10), wraplength=800)
        self.spectrum_label.pack(pady=20)
        
        # Button to update spectrum
        self.update_spectrum_button = tk.Button(spectrum_container, text="Update Eigenspectrum",
                                               command=self.update_eigenspectrum,
                                               bg="#673AB7", fg="white")
        self.update_spectrum_button.pack(pady=5)
        self.update_spectrum_button.config(state=tk.DISABLED)

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
            
            self.raw = mne.io.read_raw(filepath, preload=True, verbose=False)
            self.raw.pick_types(eeg=True, exclude=[])
            
            n_channels = len(self.raw.ch_names)
            duration = self.raw.times[-1]
            sfreq = self.raw.info['sfreq']
            self.filename = os.path.basename(filepath)
            
            info_text = (f"Loaded: {self.filename}\n"
                         f"Channels: {n_channels} | Duration: {duration:.1f}s | "
                         f"Sampling rate: {sfreq:.0f} Hz")
            self.file_info_label.config(text=info_text)
            
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
        
    def compute_harmonic_modes(self):
        """Compute harmonic modes in a separate thread"""
        self.compute_modes_button.config(state=tk.DISABLED)
        threading.Thread(target=self._compute_harmonic_modes_thread).start()
        
    def _compute_harmonic_modes_thread(self):
        """Thread function for computing harmonic modes"""
        try:
            self.update_progress(10, "Initializing harmonic engine...")
            self.update_status("Computing harmonic modes. This may take a few minutes...")
            
            # Get the current spacing from GUI
            spacing = self.spacing_var.get()
            
            # CRITICAL: Don't recreate if we already have one
            if self.harmonic_engine is None:
                self.harmonic_engine = HarmonicEngine(self.subjects_dir)
            
            # Initialize with correct spacing
            self.harmonic_engine.initialize(
                spacing=spacing,
                n_modes=self.n_modes_var.get()
            )
            
            self.harmonic_modes_computed = True
            self.update_progress(100, f"Harmonic modes computed successfully with {spacing} spacing!")
            self.log_result(f"✓ Harmonic engine initialized with {self.n_modes_var.get()} modes using {spacing} spacing.")
            
            # Add safety verification
            total_template_vertices = sum(m.shape[0] for m in self.harmonic_engine.harmonic_modes)
            expected_for_spacing = {
                'ico4': 5124,
                'ico5': 20484,
                'oct5': 40968,
                'oct6': 163848
            }
            
            if spacing in expected_for_spacing:
                expected = expected_for_spacing[spacing]
                self.log_result(f"  Expected vertices for {spacing}: {expected}")
                self.log_result(f"  Actual vertices computed: {total_template_vertices}")
                
                if total_template_vertices != expected:
                    self.log_result(f"  ⚠ WARNING: Vertex count mismatch!")
            
            # Update spectrum button state
            self.update_spectrum_button.config(state=tk.NORMAL)
            
            # Plot eigenspectrum
            self.root.after(0, self.update_eigenspectrum)
            
        except Exception as e:
            self.update_progress(0, f"Error computing harmonic modes: {str(e)}")
            self.log_result(f"✗ Error computing harmonic modes: {str(e)}")
            self.log_result(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Harmonic Engine Error", f"Failed to compute harmonic modes:\n{str(e)}")
        finally:
            self.compute_modes_button.config(state=tk.NORMAL)
            
    def update_eigenspectrum(self):
        """Update the eigenspectrum plot"""
        if self.harmonic_engine is None or not self.harmonic_modes_computed:
            messagebox.showwarning("No Harmonic Modes", "Please compute harmonic modes first.")
            return
            
        try:
            # Clear previous plot
            if self.spectrum_canvas:
                self.spectrum_canvas.get_tk_widget().destroy()
                
            # Create new figure
            self.spectrum_figure, ax = plt.subplots(figsize=(10, 6))
            
            # Plot eigenspectrum
            self.harmonic_engine.plot_eigenspectrum(ax=ax)
            
            # Embed in tkinter
            self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_figure, self.spectrum_frame)
            self.spectrum_canvas.draw()
            self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Hide the label
            self.spectrum_label.pack_forget()
            
            self.log_result("✓ Eigenspectrum updated.")
            
        except Exception as e:
            self.log_result(f"Error updating eigenspectrum: {str(e)}")
            
    def run_reconstruction(self):
        """Run source reconstruction in a separate thread"""
        self.stop_requested = False
        self.stop_button.config(state=tk.NORMAL)
        self.process_button.config(state=tk.DISABLED)
        
        self.results_text.delete(1.0, tk.END)
        self.notebook.select(self.results_frame)
        
        self.processing_thread = threading.Thread(target=self._process_data)
        self.processing_thread.start()
        
    def _process_data(self):
        """Main processing function (runs in separate thread)"""
        try:
            # Step 1: Check brain template
            self.update_progress(10, "Step 1/8: Checking brain template...")
            subjects_dir = self._check_fsaverage()
            if self.stop_requested: return
            
            # Step 2: Preprocessing
            self.update_progress(20, "Step 2/8: Preprocessing EEG data...")
            raw_processed = self._preprocess_raw()
            if self.stop_requested: return
            
            # Step 3: Create forward solution - use consistent spacing
            self.update_progress(40, "Step 3/8: Creating forward solution...")
            fwd = self._create_forward_solution(raw_processed, subjects_dir)
            if self.stop_requested: return
            
            # Get the spacing used for forward solution
            spacing = self.spacing_var.get()
            
            # Step 4: Compute inverse operator
            self.update_progress(50, "Step 4/8: Computing inverse operator...")
            inverse_operator = self._compute_inverse_operator(raw_processed, fwd)
            if self.stop_requested: return
            
            mode = self.analysis_mode.get()

            if mode == "standard":
                # Step 5: Filter for selected frequency band
                self.update_progress(60, "Step 5/8: Filtering for standard analysis...")
                raw_filtered, freq_band_name = self._filter_for_standard_analysis(raw_processed)
                if self.stop_requested: return
                
                # Step 6: Reconstruct sources
                self.update_progress(70, "Step 6/8: Reconstructing sources...")
                stc = self.reconstructor.reconstruct(raw_filtered, inverse_operator, self.method_var.get())
                self.log_result(f"Applied {self.method_var.get()} inverse solution.")

                # Step 7: Process visualization
                self.update_progress(80, "Step 7/8: Processing standard visualization...")
                stc_viz, params = self._process_standard_visualization(stc, freq_band_name)

            elif mode == "coordination":
                # Step 5: Reconstruct BROADBAND sources first
                self.update_progress(60, "Step 5/8: Reconstructing broadband sources for coordination model...")
                stc_broadband = self.reconstructor.reconstruct(raw_processed, inverse_operator, self.method_var.get())
                if self.stop_requested: return

                # Step 6: Run coordination analysis in source space
                self.update_progress(70, "Step 6/8: Calculating coordination dynamics...")
                coord_metrics = self._analyze_coordination_in_source_space(stc_broadband)
                if self.stop_requested: return

                # Step 7: Select the metric to visualize
                self.update_progress(80, "Step 7/8: Processing coordination visualization...")
                metric_to_show = self.coord_viz_var.get()
                stc_viz, params = self._process_coordination_visualization(coord_metrics, metric_to_show)
                
            elif mode == "harmonic":
                # Step 5: Reconstruct BROADBAND sources
                self.update_progress(60, "Step 5/8: Reconstructing broadband sources for harmonic analysis...")
                stc_broadband = self.reconstructor.reconstruct(raw_processed, inverse_operator, self.method_var.get())
                if self.stop_requested: return
                
                # Step 6: Get current spacing setting
                spacing = self.spacing_var.get()
                
                # CRITICAL FIX: Create or reuse harmonic engine with proper spacing check
                if self.harmonic_engine is None:
                    self.harmonic_engine = HarmonicEngine(self.subjects_dir)
                
                # Check if we need to (re)initialize
                need_recompute = (
                    not self.harmonic_modes_computed or 
                    not self.harmonic_engine.initialized or
                    getattr(self.harmonic_engine, 'spacing', None) != spacing
                )
                
                if need_recompute:
                    self.update_progress(70, f"Step 6/8: Computing harmonic modes with {spacing} spacing...")
                    # Initialize with the CORRECT spacing
                    self.harmonic_engine.initialize(spacing=spacing, 
                                                n_modes=self.n_modes_var.get())
                    self.harmonic_modes_computed = True
                    self.log_result(f"✓ Harmonic engine initialized with {self.n_modes_var.get()} modes using {spacing} spacing.")
                    
                    # Add a safety check
                    expected_vertices = sum(s['np'] for s in self.harmonic_engine.src)
                    actual_vertices = sum(m.shape[0] for m in self.harmonic_engine.harmonic_modes)
                    self.log_result(f"  Mesh check: Source space vertices = {expected_vertices}")
                    self.log_result(f"              Harmonic modes vertices = {actual_vertices}")
                    
                    if expected_vertices != actual_vertices:
                        self.log_result(f"  ⚠ WARNING: Vertex mismatch! This will cause projection errors.")
                else:
                    self.update_progress(70, "Step 6/8: Using pre-computed harmonic modes...")
                    self.log_result(f"✓ Reusing harmonic modes with {spacing} spacing.")
                
                # Step 7: Perform harmonic analysis
                self.update_progress(80, "Step 7/8: Performing harmonic decomposition...")
                stc_viz, params = self._process_harmonic_analysis(stc_broadband)
            
            # Step 8: Create visualization
            self.update_progress(95, "Step 8/8: Creating brain visualization...")
            self.root.after(0, lambda sv=stc_viz, p=params, sd=subjects_dir: 
                            self._create_visualization(sv, p, sd))
            
            self.update_progress(100, "Processing complete!")
            self.log_result("\n✓ Source reconstruction completed successfully!")
            
        except Exception as e:
            self.update_progress(0, f"Error: {str(e)}")
            self.log_result(f"\n✗ Error during processing: {str(e)}")
            self.log_result(f"Traceback: {traceback.format_exc()}")
            self.root.after(0, lambda err=str(e): messagebox.showerror("Processing Error", err))
            
        finally:
            self.process_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def _preprocess_raw(self):
        """Handles the full preprocessing pipeline."""
        raw_processed = self.raw.copy()
        
        # Clean channel names
        cleaned_names = {name: name.strip().replace('.', '').upper() for name in raw_processed.ch_names}
        raw_processed.rename_channels(cleaned_names, verbose=False)
        
        # Set montage
        montages_to_try = ['standard_1005', 'standard_1020', 'biosemi64', 'biosemi32']
        montage_set = False
        for montage_name in montages_to_try:
            try:
                montage = mne.channels.make_standard_montage(montage_name)
                raw_processed.set_montage(montage, match_case=False, on_missing='ignore', verbose=False)
                if any(ch['loc'][0] for ch in raw_processed.info['chs'] if not np.isnan(ch['loc'][0])):
                    montage_set = True
                    self.log_result(f"Successfully applied {montage_name} montage.")
                    break
            except Exception:
                continue
        if not montage_set:
            raise ValueError("Could not apply any standard montage. Please check channel names.")

        # Bad channel detection and interpolation
        if self.remove_bad_channels_var.get():
            bad_channels = self.preprocessing.detect_bad_channels(raw_processed)
            if bad_channels:
                self.log_result(f"Detected bad channels: {', '.join(bad_channels)}")
                raw_processed.info['bads'] = bad_channels
                raw_processed.interpolate_bads(reset_bads=True, verbose=False)
        
        # Artifact removal
        raw_processed = self.preprocessing.remove_artifacts(raw_processed, self.artifact_removal_var.get())
        
        # Set reference
        raw_processed.set_eeg_reference('average', projection=True, verbose=False)
        
        # Crop to selected time window
        tmin, tmax = self.time_start_var.get(), self.time_end_var.get()
        raw_processed.crop(tmin=tmin, tmax=tmax)
        self.log_result(f"Analyzing time window: {tmin}-{tmax} seconds")
        
        return raw_processed

    def _filter_for_standard_analysis(self, raw_processed):
        """Filters data for standard analysis mode."""
        freq_band_name = self.get_frequency_band()
        if freq_band_name != 'broadband':
            low_freq, high_freq = self.freq_bands[freq_band_name]
            nyquist = raw_processed.info['sfreq'] / 2.0
            if high_freq >= nyquist:
                high_freq = nyquist - 1
                self.log_result(f"Adjusted high frequency to {high_freq:.1f} Hz due to Nyquist limit.")
            raw_processed.filter(low_freq, high_freq, fir_design='firwin', verbose=False)
            self.log_result(f"Filtered to {freq_band_name} band: {low_freq}-{high_freq:.1f} Hz")
        return raw_processed, freq_band_name
    
    def _analyze_coordination_in_source_space(self, stc_broadband):
        """
        Calculates Universal Brain Coordination metrics, including the new 'Coordinated Power' (Y*PLV).
        """
        self.log_result("Analyzing coordination dynamics in source space...")
        
        conductor_band_name = self.conductor_var.get()
        conductor_freqs = self.freq_bands[conductor_band_name]
        
        orchestra_freqs = [self.freq_bands[band] for band, var in self.orchestra_vars.items() if var.get() and band != conductor_band_name]
        if not orchestra_freqs: raise ValueError("Orchestra must contain bands different from the Conductor.")
        min_orch_freq, max_orch_freq = min(f[0] for f in orchestra_freqs), max(f[1] for f in orchestra_freqs)
        
        stc_conductor = stc_broadband.copy().filter(l_freq=conductor_freqs[0], h_freq=conductor_freqs[1], verbose=False)
        stc_moire = stc_broadband.copy().filter(l_freq=min_orch_freq, h_freq=max_orch_freq, verbose=False)
        
        # Calculate intermediate data
        y_data = stc_moire.data ** 2
        phase_conductor = np.angle(hilbert(stc_conductor.data, axis=1))
        phase_moire = np.angle(hilbert(stc_moire.data, axis=1))
        
        # This is the Phase-Locking Value (PLV), which represents COORDINATION
        plv_instantaneous = np.abs(np.exp(1j * (phase_conductor - phase_moire)))
        
        # This is the Phase-Slip Rate (Z-Axis)
        z_data = 1 - plv_instantaneous
        
        # Multiply orchestra power by the coordination (PLV), not the slip rate
        coordinated_power_data = y_data * plv_instantaneous

        stc_x = mne.SourceEstimate(stc_conductor.data ** 2, vertices=stc_broadband.vertices, 
                                  tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, 
                                  subject=stc_broadband.subject)
        stc_y = mne.SourceEstimate(y_data, vertices=stc_broadband.vertices, 
                                  tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, 
                                  subject=stc_broadband.subject)
        stc_z = mne.SourceEstimate(z_data, vertices=stc_broadband.vertices, 
                                  tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, 
                                  subject=stc_broadband.subject)
        stc_cp = mne.SourceEstimate(coordinated_power_data, vertices=stc_broadband.vertices, 
                                   tmin=stc_broadband.tmin, tstep=stc_broadband.tstep, 
                                   subject=stc_broadband.subject)
        
        self.log_result("✓ Time-resolved coordination analysis complete.")
        return {
            'Conductor Power': stc_x, 
            'Moiré Harmony': stc_y, 
            'Phase-Slip Rate': stc_z, 
            'Coordinated Power (Y*PLV)': stc_cp
        }

    def _process_harmonic_analysis(self, stc):
        """Process harmonic analysis and decomposition"""
        
        # Add template disclaimer
        self.log_result("📌 Template-referenced analysis (fsaverage)")
        self.log_result("   Results reflect spatial modes of a canonical cortex")
        self.log_result("   Interpret as relative spatial scales, not exact anatomy")
        
        harmonic_type = self.harmonic_type_var.get()
        
        if harmonic_type == "mode":
            # Visualize specific harmonic mode
            mode_idx = self.mode_idx_var.get()
            if mode_idx >= self.n_modes_var.get():
                raise ValueError(f"Mode index {mode_idx} exceeds available modes ({self.n_modes_var.get()})")
            
            stc_mode = self.harmonic_engine.get_harmonic_mode_stc(mode_idx=mode_idx, 
                                                                 time_points=stc.data.shape[1])
            title = f"Harmonic Mode {mode_idx}"
            params = {
                'colormap': self.harmonic_colormap_var.get(),
                'clim': dict(kind='value', lims=[-1, 0, 1]),
                'title': title
            }
            return stc_mode, params
            
        elif harmonic_type == "spectrum":
            # Calculate harmonic power spectrum
            lh_h, rh_h = self.harmonic_engine.project_stc_to_harmonics(stc)
            
            # Calculate power in each harmonic mode (averaged over time)
            lh_power = np.mean(lh_h ** 2, axis=1)
            rh_power = np.mean(rh_h ** 2, axis=1)
            
            # Create STC with power per mode (repeated across vertices)
            n_vertices_per_hemi = self.harmonic_engine.harmonic_modes[0].shape[0]
            lh_data = np.zeros((n_vertices_per_hemi, 1))
            rh_data = np.zeros((n_vertices_per_hemi, 1))
            
            # Distribute power across vertices (simplified - each mode's power at its peak)
            for i in range(len(lh_power)):
                mode_vec = self.harmonic_engine.harmonic_modes[0][:, i]
                peak_idx = np.argmax(np.abs(mode_vec))
                lh_data[peak_idx, 0] = lh_power[i]
                
            for i in range(len(rh_power)):
                mode_vec = self.harmonic_engine.harmonic_modes[1][:, i]
                peak_idx = np.argmax(np.abs(mode_vec))
                rh_data[peak_idx, 0] = rh_power[i]
            
            # Normalize
            max_lh = np.max(lh_data) if np.max(lh_data) > 0 else 1
            max_rh = np.max(rh_data) if np.max(rh_data) > 0 else 1
            lh_data = lh_data / max_lh
            rh_data = rh_data / max_rh
            
            stc_power = mne.SourceEstimate(np.vstack([lh_data, rh_data]), 
                                          vertices=stc.vertices,
                                          tmin=stc.tmin, tstep=stc.tstep,
                                          subject=stc.subject)
            title = "Harmonic Power Spectrum"
            params = {
                'colormap': 'hot',
                'clim': dict(kind='percent', lims=[80, 90, 99]),
                'title': title
            }
            return stc_power, params
            
        else:
            # Coupled or Innovation signal
            k_cutoff = self.k_cutoff_var.get()
            coupled, innovation = self.harmonic_engine.analyze_structure_function_coupling(stc, k_cutoff)
            
            if harmonic_type == "coupled":
                lh_data, rh_data = coupled
                title = f"Structure-Coupled Signal (Modes 0-{k_cutoff-1})"
            else:  # innovation
                lh_data, rh_data = innovation
                title = f"Innovation (Decoupled) Signal (Modes {k_cutoff}+)"
            
            # Combine hemispheres
            data = np.vstack([lh_data, rh_data])
            
            # Create STC
            stc_harmonic = mne.SourceEstimate(data, vertices=stc.vertices,
                                             tmin=stc.tmin, tstep=stc.tstep,
                                             subject=stc.subject)
            
            params = {
                'colormap': self.harmonic_colormap_var.get(),
                'clim': dict(kind='percent', lims=[5, 50, 95]),
                'title': title
            }
            
            self.log_result(f"✓ Harmonic decomposition complete. Showing {harmonic_type} signal.")
            return stc_harmonic, params
    
    def _process_standard_visualization(self, stc, freq_band_name):
        """Process visualization for standard analysis"""
        viz_type = self.viz_var.get()
        title = f"{viz_type.title()} - {freq_band_name.title()} Band"
        
        stc_meta = {'vertices': stc.vertices, 'tmin': stc.tmin, 'tstep': stc.tstep, 'subject': stc.subject}

        if viz_type == "phase":
            phase_data = np.angle(hilbert(stc.data, axis=1))
            stc_viz = mne.SourceEstimate(phase_data, **stc_meta)
            params = {'colormap': 'twilight_shifted', 'clim': dict(kind='value', lims=[-np.pi, 0, np.pi]), 'title': title}
        elif viz_type == "power":
            power_data = stc.data ** 2
            stc_viz = mne.SourceEstimate(power_data, **stc_meta)
            params = {'colormap': 'hot', 'clim': dict(kind='percent', lims=[90, 95, 99]), 'title': title}
        elif viz_type == "stats":
            z_scores = (stc.data - np.mean(stc.data)) / np.std(stc.data)
            stc_viz = mne.SourceEstimate(z_scores, **stc_meta)
            params = {'colormap': 'RdBu_r', 'clim': dict(kind='value', lims=[-2.5, 0, 2.5]), 'title': title}
        else: # raw
            stc_viz = stc
            params = {'colormap': 'RdBu_r', 'clim': dict(kind='percent', lims=[5, 50, 95]), 'title': title}
        
        return stc_viz, params
    
    def _process_coordination_visualization(self, coord_metrics, metric_to_show):
        """Process visualization for coordination analysis with robust color scaling."""
        stc_viz = coord_metrics[metric_to_show]
        title = f"Coordination Model: {metric_to_show}"
        
        if metric_to_show == 'Conductor Power' or metric_to_show == 'Moiré Harmony':
            params = {
                'colormap': 'hot', 
                'clim': dict(kind='percent', lims=[90, 95, 99]), 
                'title': title
            }
        elif metric_to_show == 'Phase-Slip Rate':
            params = {
                'colormap': 'viridis', 
                'clim': dict(kind='percent', lims=[1, 50, 99]), 
                'title': title
            }
        elif 'Coordinated Power' in metric_to_show:
             params = {
                 'colormap': 'plasma',
                 'clim': dict(kind='percent', lims=[95, 97, 99.9]),
                 'title': title
             }
        
        return stc_viz, params
        
    def _create_visualization(self, stc, params, subjects_dir):
        """Create brain visualization (must be called on main thread)"""
        try:
            time_label = f"{params['title']}"
            if stc.data.ndim > 1 and stc.data.shape[1] > 1:
                 time_label += " (t=%0.2f s)"

            brain = stc.plot(
                subjects_dir=subjects_dir, subject='fsaverage', surface='pial',
                hemi='both', colormap=params['colormap'], clim=params['clim'],
                time_label=time_label, size=(1000, 750),
                smoothing_steps=5, background='white', verbose=False
            )
            self.brain_figures.append(brain)
            self.log_result(f"Created 3D visualization with {mne.viz.get_3d_backend()}")
        except Exception as e:
            self.log_result(f"3D visualization error: {str(e)}")

    def _create_forward_solution(self, raw, subjects_dir):
        """Create forward solution using the correct 3-layer BEM for EEG."""
        # Create a 3-layer BEM model suitable for EEG
        self.log_result("Creating 3-layer BEM model (brain, skull, scalp)...")
        model = mne.make_bem_model(
            subject='fsaverage', 
            ico=4,
            conductivity=(0.3, 0.006, 0.3),  # Correct 3-layer conductivity
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
        noise_cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None, verbose=False)
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, 
                                                                  loose=0.2, depth=0.8, verbose=False)
        self.log_result("Created inverse operator")
        return inverse_operator

    def _check_fsaverage(self):
        """
        Checks for the 'fsaverage' brain template and downloads it if missing.
        """
        subjects_dir = self.subjects_dir
        
        # Define the expected full path to the 'fsaverage' template directory.
        fsaverage_path = os.path.join(subjects_dir, 'fsaverage')
        
        # Check if the fsaverage directory exists. If not, download it to our defined path.
        if not os.path.isdir(fsaverage_path):
            self.log_result("Fsaverage brain template not found.")
            self.log_result(f"Downloading to: {subjects_dir} (this may take a few minutes)...")
            
            # This function will download and place the data in the correct subdirectories.
            mne.datasets.fetch_fsaverage(subjects_dir=subjects_dir, verbose=False)
            self.log_result("✓ Fsaverage download complete.")
        else:
            self.log_result("✓ Fsaverage brain template found.")

        return subjects_dir

    def get_frequency_band(self):
        """Get selected frequency band"""
        return self.freq_var.get()
    
    @property
    def freq_bands(self):
        """Frequency band definitions"""
        return {
            "delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
            "beta": (12, 30), "gamma": (30, 50), "broadband": (0.5, 50)
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

if __name__ == "__main__":
    # Set up 3D backend
    try:
        mne.viz.set_3d_backend("pyvistaqt")
    except Exception:
        try:
            mne.viz.set_3d_backend("notebook")
        except Exception:
            print("3D visualization backend not available. 2D plotting will be used.")
    
    # Create and run application
    root = tk.Tk()
    app = EEGSourceReconstructionApp(root)
    root.mainloop()