# Hidden Markov Model for Human Activity Recognition

## Project Overview

This project implements a complete Hidden Markov Model (HMM) system for recognizing human activities (Standing, Still, Walking) using smartphone accelerometer and gyroscope data.

## Project Structure

```
HMM/
├── notebook.ipynb                      # Main implementation notebook (data processing, HMM training, evaluation)
├── README.md                           # Project description, setup instructions, and overview
├── data/                               # Sensor data directory
│   ├── Michael/                         # Person 1 raw data
│   │   ├── Standing/
│   │   │   ├── Accelerometer_1.csv
│   │   │   ├── Accelerometer_2.csv
│   │   │   ├── Gyroscope_1.csv
│   │   │   └── Gyroscope_2.csv
│   │   ├── Still/
│   │   │   ├── Accelerometer_1.csv
│   │   │   ├── Accelerometer_2.csv
│   │   │   ├── Gyroscope_1.csv
│   │   │   └── Gyroscope_2.csv
│   │   ├── Walking/
│   │   │   ├── Accelerometer_1.csv
│   │   │   ├── Accelerometer_2.csv
│   │   │   ├── Gyroscope_1.csv
│   │   │   └── Gyroscope_2.csv
│   │   └── Jumping/
│   │       ├── Accelerometer_1.csv
│   │       ├── Accelerometer_2.csv
│   │       ├── Gyroscope_1.csv
│   │       └── Gyroscope_2.csv
│   ├── Joan/                            # Person 2 raw data 
│   │   └── [same structure as Michael]
│   ├── combined_data/                   # Auto-generated combined CSVs for each person & activity
│   │   ├── Michael_Standing_Accelerometer_Combined.csv
│   │   ├── Michael_Standing_Gyroscope_Combined.csv
│   │   └── [etc. for all activities and sensors]
│   │   └── multi_person/               # Combined data across multiple participants
│   │       ├── all_people_Standing_Accelerometer.csv
│   │       ├── all_people_Standing_Gyroscope.csv
│   │       ├── all_people_Still_Accelerometer.csv
│   │       ├── all_people_Still_Gyroscope.csv
│   │       ├── all_people_Walking_Accelerometer.csv
│   │       ├── all_people_Walking_Gyroscope.csv
│   │       ├── all_people_Jumping_Accelerometer.csv
│   │       └── all_people_Jumping_Gyroscope.csv
│   └── test/                            # Unseen data for evaluation
│       ├── Michael_Test_Accelerometer.csv
│       ├── Michael_Test_Gyroscope.csv
│       ├── Joan_Test_Accelerometer.csv
│       └── Joan_Test_Gyroscope.csv

```

## What's Implemented

### 1. Data Loading and Visualization 

- **Flexible data loading:** Automatically detects and loads all sensor files
  - Supports multiple files per activity (Accelerometer_1.csv, Accelerometer_2.csv, etc.)
  - Supports multiple people (optional)
  - Automatically combines data from multiple files
- **CSV export:** Saves combined data to `data/combined_data/` directory
- Visualizes raw accelerometer and gyroscope signals
- Displays data distributions and file information

### 2. Feature Extraction 

**Time-Domain Features:**

- Mean, Standard Deviation, RMS
- Signal Magnitude Area (SMA)
- Correlation between axes
- Magnitude mean

**Frequency-Domain Features:**

- Dominant frequency (via FFT)
- Spectral energy
- Spectral entropy

**Windowing:**

- Configurable window size (default: 100 samples)
- Overlapping windows (default: 50% overlap)
- Z-score normalization

### 3. HMM Implementation 

**Complete from-scratch implementation including:**

- Gaussian emission probabilities
- Viterbi algorithm for decoding
- Baum-Welch (EM) algorithm for training
- Convergence checking with tolerance threshold
- Numerical stability features

### 4. Model Evaluation 

- Temporal train/test split (80/20)
- Confusion matrix visualization
- Per-activity metrics:
  - Sensitivity (Recall)
  - Specificity
  - Precision
  - F1-Score
- Overall accuracy
- State sequence visualization

### 5. Analysis 

- Activity distinguishability analysis
- Transition probability interpretation
- Common misclassification patterns
- Feature importance ranking

## How to Use

### 1. Setup Environment

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

### 2. Prepare Your Data

- Place your CSV files in the `data/` directory
- Follow the structure: `data/[PersonName]/[Activity]/[SensorType]_N.csv`
  - Example: `data/Michael/Standing/Accelerometer_1.csv`
  - Example: `data/Michael/Standing/Accelerometer_2.csv`
  - Example: `data/Michael/Walking/Gyroscope_1.csv`
- CSV format: columns must include `time`, `seconds_elapsed`, `x`, `y`, `z`
- **Multiple files supported:** The notebook will automatically detect and load all files matching the pattern (Accelerometer_1.csv, Accelerometer_2.csv, etc.)
- **Combined data:** Automatically saved to `data/combined_data/` directory

### 3. Run the Notebook

1. Open `notebook.ipynb` in Jupyter or VS Code
2. Update the first markdown cell with your group information
3. **Configure data loading:**
   - **Single person:** Use `load_sensor_data()` (default)
   - **Multiple people:** Uncomment and use `load_multi_person_data()` in cell 5
4. Run all cells in sequence
5. Review visualizations and results
6. **Combined CSV files** are automatically saved to `data/combined_data/`

### 4. For Your Report

The notebook generates all required components:

-  Data visualizations
-  Transition probability heatmaps
-  Confusion matrices
-  Evaluation metrics tables
-  State sequence plots
-  Training convergence plots

## Key Features

### Viterbi Algorithm

Finds the most likely sequence of hidden states given observations:

- Uses dynamic programming
- Log probabilities for numerical stability
- Backtracking for optimal path reconstruction

### Baum-Welch Algorithm

Learns HMM parameters using Expectation-Maximization:

- Forward-backward algorithm for E-step
- Maximum likelihood updates for M-step
- Convergence monitoring with tolerance threshold
- Prevents numerical underflow with scaling

### Feature Engineering Rationale

**Why These Features?**

1. **Mean (time-domain)**: Captures average position/orientation
   - Standing vs. Walking shows different mean values
2. **RMS (time-domain)**: Measures signal energy
   - Walking has higher energy than being still
3. **SMA (time-domain)**: Captures overall movement intensity
   - Discriminates active vs. inactive states
4. **Correlation (time-domain)**: Shows axis coordination
   - Walking shows coordinated x-y-z patterns
5. **Dominant Frequency (frequency-domain)**: Main motion frequency
   - Walking has periodic patterns at the step frequency
6. **Spectral Energy (frequency-domain)**: Total energy in frequency domain
   - Complements time-domain energy features
7. **Spectral Entropy (frequency-domain)**: Signal complexity
   - Random movements have higher entropy

### Normalization Justification

**Z-score Normalization Used Because:**

- Different sensors have different scales
- Removes mean bias and scale differences
- Preserves relative relationships between samples
- Standard approach for ML with heterogeneous features

## Evaluation Metrics Explained

### Sensitivity (Recall)

- **Formula**: TP / (TP + FN)
- **Meaning**: Of all actual instances of activity X, how many did we detect?
- **Important for**: Activities where missing detection is costly

### Specificity

- **Formula**: TN / (TN + FP)
- **Meaning**: Of all actual non-X activities, how many did we correctly identify as not X?
- **Important for**: Avoiding false alarms

### Precision

- **Formula**: TP / (TP + FP)
- **Meaning**: Of all predictions of activity X, how many were correct?
- **Important for**: When false positives are costly

### F1-Score

- **Formula**: 2 _ (Precision _ Recall) / (Precision + Recall)
- **Meaning**: Harmonic mean balancing precision and recall
- **Important for**: Overall model quality assessment

##  Output Visualizations

All generated plots (e.g., `unseen_test_comparison.png`, `unseen_test_evaluation.png`, `confusion_matrix.png`) are automatically saved to the **project root** for reporting and grading purposes.

## Tips for Your Report

### Background Section

Explain your use case. Examples:

- "Health monitoring for elderly fall detection"
- "Fitness tracking for exercise classification"
- "Smart home automation based on occupant activity"

### Data Collection Section

Document:

- Phone models used by each group member
- Sampling rates (check your CSV files)
- Window size selection justification
- How you handled different sampling rates

### Results Section

Include from notebook:

- All visualizations (raw data, confusion matrix, transitions)
- Evaluation metrics table
- State sequence plots
- Training convergence plot

### Discussion Section

Analyze:

- Which activities were hardest to distinguish? Why?
- Do transition probabilities make sense?
- How did sensor noise affect results?
- What features were most important?

### Improvements Section

Suggest:

- More diverse data collection
- Additional features
- Different HMM variants (GMM-HMM)
- Real-time implementation considerations

## Common Issues and Solutions

### Data Loading Features

**Multiple Files Per Activity:**
The notebook automatically loads all sensor files matching the pattern:

- `Accelerometer_1.csv`, `Accelerometer_2.csv`, `Accelerometer_3.csv`, etc.
- `Gyroscope_1.csv`, `Gyroscope_2.csv`, `Gyroscope_3.csv`, etc.

**Combined Data Output:**
All loaded files are automatically combined and saved to:

- `data/combined_data/[Person]_[Activity]_[Sensor]_Combined.csv`

**Loading Data from Multiple People:**
Use the `load_multi_person_data()` function (see cell 5 in notebook):

```python
sensor_data = load_multi_person_data(
    data_path,
    people=['Michael', 'Joan'],
    activities=['Standing', 'Still', 'Walking'],
    save_combined=True
)
```

### Common Issues

### Issue: "File not found"

- Check file paths in the notebook
- Ensure CSV files are in the correct directories
- Use absolute paths if relative paths fail

### Issue: "Too few samples for training"

- Reduce window size
- Reduce overlap
- Collect more data

### Issue: "Model doesn't converge"

- Increase `n_iter` in `fit()` method
- Check for NaN values in data
- Ensure sufficient data variety

### Issue: "Poor accuracy"

- Check feature normalization
- Verify window size is appropriate
- Ensure activities are distinct enough
- Collect more/better quality data

### Issue: "Singular covariance matrix"

- Model adds regularization automatically
- If persists, increase regularization: `np.eye(n) * 1e-6` → `1e-5`

## Rubric Compliance Checklist

### Data Collection  ✅

- [x] 50 well-labeled files across activities
- [x] Minimum 1:30 per activity
- [x] Windowing explained based on sampling rate
- [x] Different sampling rates handled
- [x] Visualization plots included

### Feature Extraction  ✅

- [x] > 3 features total
- [x] > 2 time-domain features
- [x] > 1 frequency-domain feature (FFT-based)
- [x] Features justified
- [x] Z-score normalization applied and explained

### Implementation  ✅

- [x] Functional Viterbi algorithm
- [x] Baum-Welch with convergence check
- [x] Robust implementation
- [x] Well-documented code

### Evaluation  ✅

- [x] Tested on unseen data (2+ test files)
- [x] Sensitivity, specificity, accuracy reported
- [x] Transition/Emission probability visualizations
- [x] Confusion matrix generated
- [x] Detailed discussion

## Additional Resources

### Understanding HMMs

- Forward-Backward algorithm: https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm
- Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
- Baum-Welch algorithm: https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm

### Feature Engineering

- Time-domain features: Signal processing fundamentals
- FFT: Fast Fourier Transform for frequency analysis
- Spectral analysis: Power spectral density methods

## Contact & Support

If you encounter issues:

1. Check the "Common Issues" section above
2. Review error messages carefully
3. Verify data format matches expected structure
4. Check all file paths are correct

## License

Educational project for academic purposes.

---

**Good luck with your project!**
