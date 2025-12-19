# EEG Brain Source Reconstruction Tool (Enhanced Version)

EDIT: Added new version that: "This code performs EEG source reconstruction on a canonical cortical surface and decomposes the resulting activity into connectome harmonic modes derived from the cortical Laplacian. It separates structure-coupled (low-order) activity from structure-decoupled “innovation” signals (high-order modes), enabling time-resolved visualization and analysis of structure–function decoupling in the human brain." Naturally as of now using the generic MNE brain so the EEG wont fit the brain. 

(Vibecoded - new version by Opus) 

A comprehensive GUI tool for EEG source localization using MNE-Python. This enhanced version features multiple inverse methods, advanced preprocessing, and robust visualization options.

![EEG Source Reconstruction](brain.png)

## ⚠️ Important Notice

**This is an educational/research tool. Not for medical use.** Results are computational estimates based on mathematical models, not actual brain imaging. Always consult qualified medical professionals for health-related concerns.

## 🚀 New Features in Enhanced Version

- **Multiple inverse methods**: sLORETA, dSPM, MNE, eLORETA
- **Advanced preprocessing**: Automatic bad channel detection, artifact removal (basic & ICA)
- **Flexible frequency analysis**: Including broadband (0.5-50 Hz) option
- **Enhanced visualizations**: Power, phase, raw amplitude, and statistical maps
- **Better UI**: Tabbed interface, progress bar, real-time logging
- **Robust error handling**: Automatic montage matching, Nyquist frequency checking
- **Export capabilities**: Save source estimates and figures

## 🧠 What This Tool Does

- **Loads EEG files** (EDF, BDF, FIF, SET, VHDR formats)
- **Preprocesses data** automatically (filtering, artifact removal, bad channel interpolation)
- **Reconstructs brain sources** using multiple inverse modeling methods
- **Visualizes results** on interactive 3D brain or 2D fallback plots
- **Analyzes frequency bands** (Delta, Theta, Alpha, Beta, Gamma, Broadband)
- **Shows various metrics** (Phase synchronization, Power distribution, Statistical significance)

## ❌ What This Tool Does NOT Do

- **Not a medical device or diagnostic tool**
- **Does not create real brain scans (like MRI/CT)**
- **Cannot replace clinical EEG interpretation**
- **Results are estimates, not ground truth**

## 🔧 Technical Improvements

### Preprocessing Pipeline
- Automatic bad channel detection using variance-based z-scores
- Keeps important frontal channels (FP1, FP2) despite high variance
- ICA-based EOG artifact removal option
- Smart resampling based on frequency band requirements
- Proper handling of line noise (50/60 Hz notch filtering)

### Source Reconstruction Methods
1. **sLORETA** - Standardized low-resolution electromagnetic tomography (default)
2. **dSPM** - Dynamic statistical parametric mapping
3. **MNE** - Minimum norm estimate
4. **eLORETA** - Exact low-resolution electromagnetic tomography

### Visualization Types
- **Power Distribution**: Shows strength of brain activity
- **Phase Patterns**: Reveals timing/synchronization of brain waves
- **Raw Amplitude**: Direct reconstructed source signals
- **Statistical Maps**: Z-score thresholded significant activations

## 📋 Requirements

```bash
# Core dependencies
pip install mne numpy scipy matplotlib tkinter

# For 3D visualization (highly recommended)
pip install pyvista pyvistaqt

# Optional: For ICA artifact removal
pip install scikit-learn

# Optional: Alternative 3D backend
pip install mayavi
```

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/EEGBrainSourceReconstructionTool
cd EEGBrainSourceReconstructionTool
pip install -r requirements.txt
python EEGBrainSourceReconstructionTool2.py
```

## 📖 Usage Guide

### Quick Start
1. **Launch the application**
2. **Load EEG file** using "Load EEG File" button
3. **Configure settings** in Basic/Advanced tabs
4. **Click "Process & Reconstruct"**
5. **Interact with 3D brain** visualization

### Basic Settings
- **Frequency Band**: Choose the brain rhythm to analyze
  - Delta (0.5-4 Hz): Deep sleep, unconscious processes
  - Theta (4-8 Hz): Memory, meditation, creativity
  - Alpha (8-12 Hz): Relaxed awareness, eyes closed
  - Beta (12-30 Hz): Active thinking, focus
  - Gamma (30-50 Hz): Cognitive processing, binding
  - Broadband (0.5-50 Hz): All frequencies

- **Visualization Type**: How to display the results
  - Power: Best for seeing active brain regions
  - Phase: Shows synchronization patterns
  - Raw: Direct source amplitude
  - Statistical: Highlights significant activations

### Advanced Settings
- **Preprocessing Options**:
  - Auto-detect bad channels
  - Artifact removal (basic filters or ICA)
  
- **Source Reconstruction**:
  - Choose inverse method
  - Adjust source space density
  - Set SNR/regularization

- **Output Options**:
  - Save source estimates
  - Export figures

### 3D Brain Controls
- **Left click + drag**: Rotate brain
- **Right click + drag**: Zoom in/out
- **Middle click + drag**: Pan view
- **Spacebar**: Start/stop time animation
- **Time slider**: Explore different time points

## 📊 File Requirements

- **Multi-channel EEG** (minimum 10 channels recommended)
- **Standard electrode names** (10-20, 10-10, or 10-5 system)
- **Supported formats**: 
  - European Data Format (.edf)
  - BioSemi Data Format (.bdf)
  - MNE native format (.fif)
  - EEGLAB format (.set)
  - BrainVision format (.vhdr)

## 🧪 Scientific Background

This tool implements peer-reviewed methods from computational neuroscience:

### Forward Modeling
- **BEM (Boundary Element Method)**: 3-layer realistic head model
- **Conductivity values**: Brain (0.3), Skull (0.006), Scalp (0.3) S/m
- **Source space**: Cortical surface with adjustable density

### Inverse Solutions
- **sLORETA**: Zero localization error in noise-free simulations
- **dSPM**: Provides statistical maps with noise normalization
- **MNE**: L2 minimum norm with depth weighting
- **eLORETA**: Exact zero-error localization in the presence of measurement noise

### Brain Template
- **fsaverage**: FreeSurfer average brain from 40 subjects
- **Surface models**: Pial and white matter surfaces
- **Coordinate system**: MNI space

## ⚠️ Known Limitations

1. **Spatial Resolution**: ~1-2 cm at best (inherent to EEG)
2. **Deep Sources**: Limited sensitivity to subcortical structures
3. **Nyquist Frequency**: High-frequency analysis limited by sampling rate
4. **Head Model**: Uses template brain, not individual anatomy
5. **Temporal Smoothing**: Some methods assume temporal stationarity

## 🔍 Troubleshooting

### Common Issues

**"Cannot fit headshape without digitization"**
- Solution: Tool now automatically applies standard montages
- Ensure channel names follow standard conventions

**"Nyquist frequency error"**
- Solution: Tool automatically adjusts filtering based on sampling rate
- For gamma band, need at least 100 Hz sampling rate

**"3D visualization not appearing"**
- Solution: Fixed in enhanced version with proper threading
- Fallback to 2D plots if 3D backends unavailable

**"Bad channels detected"**
- Solution: Tool handles automatically, keeps important frontal channels
- Can disable in Advanced Settings if needed

### Platform-Specific Notes

**Windows**: Qt warnings are cosmetic and can be ignored
**macOS**: May need to install XQuartz for some 3D backends
**Linux**: Ensure OpenGL drivers are properly installed

## 📚 Educational Resources

- [MNE-Python Tutorials](https://mne.tools/stable/tutorials/)
- [Source Localization Theory](https://www.sciencedirect.com/topics/neuroscience/source-localization)
- [EEG Forward/Inverse Problem](https://en.wikipedia.org/wiki/EEG_analysis#Source_localization)

## 📄 License

MIT License - Free for educational and research use

## 🙏 Acknowledgments

- **MNE-Python developers** for the comprehensive neuroimaging toolkit
- **FreeSurfer team** for the fsaverage brain template
- **PyVista developers** for 3D visualization capabilities
- **Neuroscience community** for method development and validation

**Remember**: This tool provides computational estimates of brain activity based on scalp EEG. Results should be interpreted with caution and in the context of the data quality, preprocessing steps, and inherent limitations of EEG source localization.
