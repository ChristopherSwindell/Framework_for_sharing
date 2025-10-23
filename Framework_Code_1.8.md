## Workflow

1. Generate datasets
2. Train MLP models
3. Run sample selection code
4. Run SHAP analysis
5. Calculate DCG comparisons and statistical tests

## Group 1
For group 1, I am comparing 4 SHAP variations using raw (not normalized) data. I hypothesize the SHAP will perform better on spatial data when background data is calculated just for the local area (queen configuration). 

For background, I use normalized data when considering LIME so that I can directly convert variables to feature importance lists using the absolute values of the coefficient estimates. This will occur in group 2. (Group numbers are arbitrary and just reflect the order in which I am running the code.) However, normalizing the data removes the ability to compare whether using global background or queen (configuration) background data performs better for SHAP.

For group 1, I will
* Create five datasets representing different types of spatial configurations (2-D plane) using 8 variables (8-D feature space)
* Train models on unnormalized data
* Compare 4 SHAP variations using raw data:
   1. Global background + Decision-aligned
   2. Queen background + Decision-aligned
   3. Global background + Class-specific
   4. Queen background + Class-specific

### Spatial Pattern Generation

Datasets include 8 variables, a 10 X 10 grid where each cell is considered a region of interest (ROI), and each ROI contains 100 samples.

In this code, I create five spatial patterns:
1. Smooth gradient - here we see a gradual change in variable importance across the 10 by 10 grid. It reflects the idea that nearer things are more alike than more distant things or spatial decay.
2. Sharp boundary - here we see a defined change in the variable importance at a particular boundary point as we might see when an interstate or river cuts through a city or the landscape is marked by sudden changes in elevation (e.g. mountains)
3. Hotspot clustering - here see a cluster where some variables have markedly more importance than in other areas. We might see this pattern with areas of a city devoted to industrial processes vs retail vs residential or with cultural associations (e.g., a China Town).
4. Random - here there is no defined spatial organization.

As a check that the spatial patterns are generated as intended, visualizations are created and two statistical methods of spatial autocorrelation (Moran's I and Geary's C) are calculated.

### Global class boundaries
Global class boundaries are created because:
1. Using fixed thresholds across all ROIs and datasets ensures that class labels reflect absolute spatial variation, not just local rank.
2. Fixed thresholds reduce label noise and class imbalance across ROIs, which helps the MLP converge more reliably and generalize better.
3. It permits the comparison of performance metrics (e.g., accuracy, precision) across ROIs or datasets without worrying about shifting class definitions
4. It potentially improves explainer reliability, e.g., SHAP explanations are more meaningful when the target variable has consistent semantics.

# Complete Pipeline Structure:

## Part 1: Spatial Autocorrelation Analyzer

* Calculates Moran's I and Geary's C

## Part 2A: Standard Dataset Generator (8 features)

* SpatialPatternGenerator - For Group 1 (SHAP + LIME)
* Generates raw data without spatial coordinates
* All 4 dataset types (smooth gradient, sharp boundary, clustered hotspots, random)

## Part 2B: Geospatial Dataset Generator (10 features)

* GeoSpatialPatternGenerator - For Group 2 (SHAP + GeoSHAP)
* Automatically adds normalized spatial coordinates (x, y) as features 8 and 9
* Returns both X (10 features) and X_original (8 features for ground truth)
* Updated clustering to have 4 hotspots (one per quadrant)

## Part 3: High-Level Workflow Functions

* generate_datasets_group_1() - Creates 8-variable datasets
* generate_datasets_group_2() - Creates 10-variable datasets with spatial coords
* visualize_all_datasets() - Batch visualization
* analyze_spatial_statistics() - Batch spatial stats calculation
* select_samples_for_xai() - Sample selection for both groups

## Part 4: Table Generation

* Generates both groups
* Runs complete pipeline
* Provides clear summary


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import pandas as pd
from scipy import stats as scipy_stats

# Set seed for reproducibility
np.random.seed(42)
```

## PART 1: SPATIAL AUTOCORRELATION ANALYZER


```python
class SpatialAutocorrelationAnalyzer:
    """Calculate spatial autocorrelation statistics for grid-based data"""
    
    @staticmethod
    def create_spatial_weights(grid_size, weight_type='queen'):
        """
        Create spatial weights matrix for a grid
        
        Parameters:
        - grid_size: Size of the grid (grid_size x grid_size)
        - weight_type: 'queen' (8 neighbors) or 'rook' (4 neighbors)
        
        Returns:
        - W: Spatial weights matrix (row-normalized)
        """
        n = grid_size * grid_size
        W = np.zeros((n, n))
        
        for i in range(grid_size):
            for j in range(grid_size):
                roi_idx = i * grid_size + j
                neighbors = []
                
                # Define neighbor offsets
                if weight_type == 'queen':
                    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
                elif weight_type == 'rook':
                    offsets = [(-1,0), (0,-1), (0,1), (1,0)]
                
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        neighbor_idx = ni * grid_size + nj
                        neighbors.append(neighbor_idx)
                
                # Set weights (equal weight to all neighbors)
                if neighbors:
                    for neighbor_idx in neighbors:
                        W[roi_idx, neighbor_idx] = 1.0 / len(neighbors)
        
        return W
    
    @staticmethod
    def morans_i(values, W):
        """Calculate Global Moran's I statistic"""
        n = len(values)
        W_sum = np.sum(W)
        
        if W_sum == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        # Center the values
        y_mean = np.mean(values)
        y_centered = values - y_mean
        
        # Calculate Moran's I
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += W[i,j] * y_centered[i] * y_centered[j]
        
        denominator = np.sum(y_centered**2)
        
        if denominator == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        
        I = (n / W_sum) * (numerator / denominator)
        
        # Calculate expected value and variance
        expected_I = -1 / (n - 1)
        
        # Simplified variance calculation
        S1 = 0.5 * np.sum((W + W.T)**2)
        S2 = np.sum((np.sum(W, axis=1) + np.sum(W, axis=0))**2)
        
        b2 = n * np.sum(y_centered**4) / (np.sum(y_centered**2)**2)
        
        variance_I = ((n * S1 - n * S2 + 3 * W_sum**2) / 
                     ((n - 1) * (n - 2) * (n - 3) * W_sum**2)) - \
                     ((b2 - n) / ((n - 1) * (n - 2) * (n - 3))) - expected_I**2
        
        # Calculate z-score and p-value
        if variance_I > 0:
            z_score = (I - expected_I) / np.sqrt(variance_I)
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
        else:
            z_score = np.nan
            p_value = np.nan
        
        return I, expected_I, variance_I, z_score, p_value
    
    @staticmethod
    def gearys_c(values, W):
        """Calculate Geary's C statistic"""
        n = len(values)
        W_sum = np.sum(W)
        
        if W_sum == 0:
            return np.nan, 1.0, np.nan, np.nan
        
        # Calculate Geary's C
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += W[i,j] * (values[i] - values[j])**2
        
        y_mean = np.mean(values)
        denominator = 2 * np.sum((values - y_mean)**2)
        
        if denominator == 0:
            return np.nan, 1.0, np.nan, np.nan
        
        C = ((n - 1) / W_sum) * (numerator / denominator)
        
        expected_C = 1.0
        
        # Simplified variance calculation for Geary's C
        S1 = 0.5 * np.sum((W + W.T)**2)
        variance_C = ((2 * S1 + S1 * (n - 1)) / ((n + 1) * W_sum**2))
        
        # Calculate z-score and p-value
        if variance_C > 0:
            z_score = (C - expected_C) / np.sqrt(variance_C)
            p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))
        else:
            z_score = np.nan
            p_value = np.nan
        
        return C, expected_C, z_score, p_value
```

## PART 2: DATASET GENERATOR CLASS

### Part 2A: Spatial dataset without coordinates


```python
class SpatialPatternGenerator:
    """Generate synthetic spatial datasets with various spatial patterns"""
    
    def __init__(self, grid_size=10, n_features=8, n_samples_per_roi=100):
        """
        Initialize the spatial pattern generator
        
        Parameters:
        - grid_size: Size of spatial grid (grid_size x grid_size ROIs)
        - n_features: Number of features (8 for SHAP+LIME, 10 for SHAP+GeoSHAP)
        - n_samples_per_roi: Number of data points per ROI
        """
        self.grid_size = grid_size
        self.n_features = n_features
        self.n_samples_per_roi = n_samples_per_roi
        self.n_rois = grid_size * grid_size
        
        # Create coordinate system
        self.coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    
    # ------------------------------------------------------------------------
    # Dataset Generation Methods
    # ------------------------------------------------------------------------
    
    def generate_dataset_a_smooth_gradient(self, noise_level=0.25, normalize_data=True, global_boundaries=None):
        """
        Dataset A: Smooth gradient with gradual spatial transitions
        Features vary smoothly across space following diagonal gradients
        """
        print("Generating Dataset A: Smooth Spatial Gradients")
        
        X_all, roi_labels = [], []
        roi_feature_weights = {}
        
        for roi_idx, (row, col) in enumerate(self.coords):
            # Normalized coordinates in [0,1]
            r = row / (self.grid_size - 1)
            c = col / (self.grid_size - 1)
            diag_from_tl = (row + col) / (2 * (self.grid_size - 1))
            diag_from_br = ((self.grid_size - 1 - row) + (self.grid_size - 1 - col)) / (2 * (self.grid_size - 1))
            
            # Create feature means following smooth gradients
            feature_means = np.zeros(self.n_features)
            
            # Features 0-2: gradient from top-left (high at TL, low at BR)
            feature_means[0] = 8.0 * (1 - diag_from_tl)
            feature_means[1] = 6.0 * (1 - diag_from_tl)
            feature_means[2] = 4.0 * (1 - diag_from_tl)
            
            # Features 3-5: gradient from bottom-right (high at BR, low at TL)
            feature_means[3] = 8.0 * (1 - diag_from_br)
            feature_means[4] = 6.0 * (1 - diag_from_br)
            feature_means[5] = 4.0 * (1 - diag_from_br)
            
            # Features 6-7: simple row/column gradients
            feature_means[6] = 6.0 * r
            feature_means[7] = 6.0 * c
            
            # For 10-variable datasets, add two more features with different patterns
            if self.n_features == 10:
                feature_means[8] = 5.0 * (r + c) / 2  # Average gradient
                feature_means[9] = 5.0 * np.abs(r - c)  # Difference gradient
            
            roi_feature_weights[roi_idx] = feature_means.copy()
            
            # Generate samples with constant noise
            X_roi = np.random.normal(
                loc=feature_means, 
                scale=noise_level, 
                size=(self.n_samples_per_roi, self.n_features)
            )
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)

        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)
        
        # Generate target labels with custom approach for smooth gradient
        if global_boundaries is not None:
            # Ignore global boundaries and create local ones for better class balance
            beta = np.array([1.0, 0.8, 0.6, -0.5, -0.3, -0.1, 0.4, 0.4])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        else:
            beta = np.array([1.0, 0.8, 0.6, -0.5, -0.3, -0.1, 0.4, 0.4])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        
        return self._package_dataset(X, y, roi_labels, "smooth_gradient", roi_feature_weights, normalize_data)
            
    def generate_dataset_b_sharp_boundary(self, noise_level=0.05, normalize_data=True, global_boundaries=None):
        """
        Dataset B: Sharp horizontal boundary creating distinct north/south regions
        Features alternate in importance pattern between regions
        """
        print("Generating Dataset B: Sharp Boundary Effect")
        
        boundary_row = self.grid_size // 2
        X_all, roi_labels = [], []
        roi_feature_weights = {}
        
        for roi_idx, (row, col) in enumerate(self.coords):
            # Base feature template
            base_features = np.array([4.0, 4.0, 3.5, 3.5, 3.0, 3.0, 2.5, 2.5])
            
            if row < boundary_row:  # North region
                multipliers = np.array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
            else:  # South region
                multipliers = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
            
            feature_means = base_features * multipliers
            
            # For 10-variable datasets
            if self.n_features == 10:
                if row < boundary_row:
                    feature_means = np.append(feature_means, [5.0, 2.0])
                else:
                    feature_means = np.append(feature_means, [2.0, 5.0])
            
            roi_feature_weights[roi_idx] = feature_means.copy()
            
            # Generate samples with constant noise
            X_roi = np.random.normal(
                loc=feature_means,
                scale=noise_level,
                size=(self.n_samples_per_roi, self.n_features)
            )
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)
        
        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)
        
        # Generate target labels - use custom beta for sharp boundary
        if global_boundaries is not None:
            # For sharp boundary, ignore global boundaries and create local ones
            # This pattern needs different weights to create balanced classes
            beta = np.array([1.0, -1.0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        else:
            beta = np.array([1.0, -1.0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        
        return self._package_dataset(X, y, roi_labels, "sharp_boundary", roi_feature_weights, normalize_data)
    
    def generate_dataset_c_clustered_hotspots(self, noise_level=0.05, normalize_data=True, global_boundaries=None):
        """
        Dataset C: Four clustered hotspots (one in each quadrant)
        Each hotspot emphasizes different feature pairs
        """
        print("Generating Dataset C: Clustered Hotspots Pattern (4 quadrants)")
        
        X_all, roi_labels = [], []
        roi_feature_weights = {}
        
        # Define hotspot centers for each quadrant
        g = self.grid_size
        hotspot_centers = [
            np.array([int(0.25 * (g - 1)), int(0.25 * (g - 1))]),  # Quadrant 1: top-left
            np.array([int(0.25 * (g - 1)), int(0.75 * (g - 1))]),  # Quadrant 2: top-right
            np.array([int(0.75 * (g - 1)), int(0.25 * (g - 1))]),  # Quadrant 3: bottom-left
            np.array([int(0.75 * (g - 1)), int(0.75 * (g - 1))]),  # Quadrant 4: bottom-right
        ]
        
        # Hotspot parameters
        radius = max(2.0, 0.3 * (g - 1))
        intensity = 8.0
        crosstalk = 0.10
        baseline = 0.30
        
        # Dominant feature pairs for each hotspot
        dominant_features = [
            [0, 1],  # Hotspot 1
            [2, 3],  # Hotspot 2
            [4, 5],  # Hotspot 3
            [6, 7],  # Hotspot 4
        ]
        
        # For 10-feature datasets, adjust last hotspot
        if self.n_features == 10:
            dominant_features[3] = [8, 9]

        # Generate ROI data
        for roi_idx, (row, col) in enumerate(self.coords):
            coord = np.array([row, col], dtype=float)
            
            # Start with low baseline
            feature_means = np.full(self.n_features, baseline, dtype=float)
            
            # Add influence from each hotspot (Gaussian decay)
            for center, dom_features in zip(hotspot_centers, dominant_features):
                dist = np.linalg.norm(coord - center)
                
                if dist <= radius:
                    influence = intensity * np.exp(-(dist ** 2) / (radius ** 2))
                    
                    # Strong bump for dominant features
                    feature_means[dom_features] += influence
                    
                    # Mild spillover for other features
                    other_features = [k for k in range(self.n_features) if k not in dom_features]
                    feature_means[other_features] += crosstalk * influence
            # Keep everything positive
            feature_means = np.clip(feature_means, 0.05, None)
            
            roi_feature_weights[roi_idx] = feature_means.copy()
            
            # Generate samples
            X_roi = np.random.normal(
                loc=feature_means,
                scale=noise_level,
                size=(self.n_samples_per_roi, self.n_features)
            )
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)
        
        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)
        
        # Use alternating positive/negative weights for better separation
        if global_boundaries is not None:
            beta = np.array([1.0, -0.8, 0.9, -0.7, 0.8, -0.6, 0.7, -0.5])

        # Use alternating positive/negative weights for better separation
        if global_boundaries is not None:
            beta = np.array([1.0, -0.8, 0.9, -0.7, 0.8, -0.6, 0.7, -0.5])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        else:
            beta = np.array([1.0, -0.8, 0.9, -0.7, 0.8, -0.6, 0.7, -0.5])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        
        return self._package_dataset(X, y, roi_labels, "clustered_hotspots", roi_feature_weights, normalize_data)
    
    def generate_dataset_d_random_pattern(self, noise_level=0.05, normalize_data=True, global_boundaries=None):
        """
        Dataset D: Random pattern (negative control)
        No spatial structure - each ROI has independent random feature weights
        """
        print("Generating Dataset D: Random Pattern (Negative Control)")
        
        X_all, roi_labels = [], []
        roi_feature_weights = {}
        
        for roi_idx, coord in enumerate(self.coords):
            # Random feature weights for this ROI
            feature_weights = np.random.uniform(0.1, 8.0, self.n_features)
            roi_feature_weights[roi_idx] = feature_weights.copy()
            
            # Generate samples
            X_roi = np.random.randn(self.n_samples_per_roi, self.n_features)
            
            # Scale by feature weights
            for i in range(self.n_features):
                X_roi[:, i] *= feature_weights[i]
            
            # Add noise
            X_roi += np.random.normal(0, noise_level, X_roi.shape)
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)
        
        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)
        
        # Generate target labels
        y = self._generate_target_labels(X, global_boundaries, dataset_type='random')
        
        return self._package_dataset(X, y, roi_labels, "random_pattern", roi_feature_weights, normalize_data)
    
    # ------------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------------
    
    def _generate_target_labels(self, X, global_boundaries, dataset_type):
        """Generate target labels using either global or local boundaries"""
        
        if global_boundaries is not None:
            # Use provided global boundaries
            global_p33, global_p67, beta = global_boundaries
            weighted_sum = X @ beta
            y = np.digitize(weighted_sum, bins=[global_p33, global_p67])
        else:
            # Calculate local boundaries based on dataset type
            if self.n_features == 8:
                if dataset_type == 'gradient':
                    beta = np.array([1.0, 0.8, 0.6, 1.0, 0.8, 0.6, 0.7, 0.7])
                elif dataset_type == 'boundary':
                    beta = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5])
                elif dataset_type == 'hotspots':
                    beta = np.array([1.0, 0.9, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9])
                else:  # random
                    beta = np.ones(8)
            else:  # 10 features
                if dataset_type == 'gradient':
                    beta = np.array([1.0, 0.8, 0.6, 1.0, 0.8, 0.6, 0.7, 0.7, 0.6, 0.5])
                elif dataset_type == 'boundary':
                    beta = np.array([1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 0.8, 0.8])
                elif dataset_type == 'hotspots':
                    beta = np.array([1.0, 0.9, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
                else:  # random
                    beta = np.ones(10)
            
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        
        return y
    
    def _package_dataset(self, X, y, roi_labels, dataset_name, roi_feature_weights, normalize_data):
        """Package dataset with optional normalization"""
        
        # Always store raw data for spatial statistics
        X_raw = X.copy()
        
        if normalize_data:
            # ROI-wise normalization
            X_normalized = X.copy()
            roi_scalers = {}
            
            for roi_idx in range(self.n_rois):
                roi_mask = roi_labels == roi_idx
                n_roi_samples = np.sum(roi_mask)
                
                if n_roi_samples > 1:
                    roi_scaler = StandardScaler()
                    X_normalized[roi_mask] = roi_scaler.fit_transform(X[roi_mask])
                    roi_scalers[roi_idx] = roi_scaler
                elif n_roi_samples == 1:
                    # Single sample: just center it
                    roi_sample = X[roi_mask]
                    X_normalized[roi_mask] = roi_sample - np.mean(roi_sample, axis=0)
                    roi_scalers[roi_idx] = None
            
            return {
                'X': X_normalized,
                'X_raw': X_raw,
                'y': y,
                'roi_labels': roi_labels,
                'coords': self.coords,
                'grid_size': self.grid_size,
                'n_features': self.n_features,
                'dataset_name': dataset_name,
                'scaler': None,
                'roi_scalers': roi_scalers,
                'roi_feature_weights': roi_feature_weights
            }
        else:
            # Return raw data
            return {
                'X': X_raw,
                'X_raw': X_raw,
                'y': y,
                'roi_labels': roi_labels,
                'coords': self.coords,
                'grid_size': self.grid_size,
                'n_features': self.n_features,
                'dataset_name': dataset_name,
                'scaler': None,
                'roi_scalers': None,
                'roi_feature_weights': roi_feature_weights
            }
    
    def select_consistent_samples(self, dataset_dict, random_state=42):
        """
        Select one representative sample per ROI for XAI analysis
        
        Parameters:
        - dataset_dict: Dataset dictionary from generate_dataset_* methods
        - random_state: Random seed for reproducibility
        
        Returns:
        - Dictionary with selected samples (100 samples total, 1 per ROI)
        """
        np.random.seed(random_state)
        
        X = dataset_dict['X']
        y = dataset_dict['y']
        roi_labels = dataset_dict['roi_labels']
        
        selected_indices = []
        
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            roi_indices = np.where(roi_mask)[0]
            
            if len(roi_indices) > 0:
                # Select random sample from this ROI
                selected_idx = np.random.choice(roi_indices)
                selected_indices.append(selected_idx)
        
        selected_indices = np.array(selected_indices)
        
        return {
            'indices': selected_indices,
            'X': X[selected_indices],
            'y': y[selected_indices],
            'roi_labels': np.arange(self.n_rois),  # One sample per ROI
            'dataset_name': dataset_dict['dataset_name'] + '_selected'
        }
    
    # ------------------------------------------------------------------------
    # Visualization and Analysis Methods
    # ------------------------------------------------------------------------
    
    def visualize_roi_grid(self, dataset_dict, feature_idx=0, show_values=True):
        """Visualize spatial pattern for a specific feature"""
        coords = dataset_dict['coords']
        X = dataset_dict.get('X_raw', dataset_dict['X'])
        roi_labels = dataset_dict['roi_labels']
        
        # Calculate mean feature value for each ROI
        roi_means = []
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            roi_mean = np.mean(X[roi_mask, feature_idx])
            roi_means.append(roi_mean)
        
        # Reshape to grid
        grid_values = np.array(roi_means).reshape(self.grid_size, self.grid_size)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(grid_values, cmap='RdYlBu_r', origin='lower')
        plt.colorbar(im, label=f'Mean Feature {feature_idx} Value', shrink=0.8)
        plt.title(f'{dataset_dict["dataset_name"].replace("_", " ").title()} - Feature {feature_idx}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Grid X Coordinate', fontsize=12)
        plt.ylabel('Grid Y Coordinate', fontsize=12)
        
        # Add grid lines
        for i in range(self.grid_size + 1):
            plt.axhline(i - 0.5, color='white', linewidth=1, alpha=0.7)
            plt.axvline(i - 0.5, color='white', linewidth=1, alpha=0.7)
        
        # Add value annotations
        if show_values:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    value = grid_values[i, j]
                    text_color = 'white' if value > np.mean(grid_values) else 'black'
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=text_color, fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def calculate_spatial_statistics(self, dataset_dict, feature_indices=None):
        """Calculate spatial autocorrelation statistics"""
        if feature_indices is None:
            feature_indices = list(range(self.n_features))
        
        roi_labels = dataset_dict['roi_labels']
        X = dataset_dict.get('X_raw', dataset_dict['X'])
        dataset_name = dataset_dict['dataset_name']
        
        # Create spatial weights matrix
        W = SpatialAutocorrelationAnalyzer.create_spatial_weights(self.grid_size, 'queen')
        
        results = {
            'dataset_name': dataset_name,
            'spatial_weights_type': 'queen',
            'features_analyzed': feature_indices,
            'statistics': {}
        }
        
        print(f"\nCalculating spatial autocorrelation for {dataset_name}...")
        
        for feature_idx in feature_indices:
            print(f"  Feature {feature_idx}:", end=" ")
            
            # Calculate mean feature value for each ROI
            roi_means = []
            for roi_idx in range(self.n_rois):
                roi_mask = roi_labels == roi_idx
                roi_mean = np.mean(X[roi_mask, feature_idx])
                roi_means.append(roi_mean)
            
            roi_means = np.array(roi_means)
            
            # Calculate Moran's I
            I, expected_I, var_I, z_I, p_I = SpatialAutocorrelationAnalyzer.morans_i(roi_means, W)
            
            # Calculate Geary's C
            C, expected_C, z_C, p_C = SpatialAutocorrelationAnalyzer.gearys_c(roi_means, W)
            
            results['statistics'][feature_idx] = {
                'roi_means': roi_means,
                'morans_i': {
                    'statistic': I,
                    'expected': expected_I,
                    'variance': var_I,
                    'z_score': z_I,
                    'p_value': p_I,
                    'significant': p_I < 0.05 if not np.isnan(p_I) else False
                },
                'gearys_c': {
                    'statistic': C,
                    'expected': expected_C,
                    'z_score': z_C,
                    'p_value': p_C,
                    'significant': p_C < 0.05 if not np.isnan(p_C) else False
                }
            }
            
            # Print summary
            if not np.isnan(I):
                significance = "***" if p_I < 0.001 else "**" if p_I < 0.01 else "*" if p_I < 0.05 else "ns"
                print(f"Moran's I = {I:.3f} {significance}, Geary's C = {C:.3f}")
            else:
                print("No variation (uniform values)")
        
        return results
```

### Part 2B: Spatial dataset with coordinates


```python
class GeoSpatialPatternGenerator:
    """
    Generate synthetic spatial datasets WITH explicit spatial coordinate features
    for geoSHAP analysis
    """
    
    def __init__(self, grid_size=10, n_features=8, n_samples_per_roi=100):
        """
        Generate synthetic spatial datasets with spatial coordinates
        
        Parameters:
        - grid_size: Size of spatial grid (grid_size x grid_size ROIs)
        - n_features: Number of non-spatial features (spatial coords added separately)
        - n_samples_per_roi: Number of data points per ROI
        """
        self.grid_size = grid_size
        self.n_features = n_features  # Non-spatial features
        self.n_samples_per_roi = n_samples_per_roi
        self.n_rois = grid_size * grid_size
        
        # Create coordinate system (will be added as features)
        self.coords = np.array([(i, j) for i in range(grid_size) for j in range(grid_size)])
    
    def _add_spatial_features(self, X, roi_labels):
        """
        Add normalized spatial coordinates as explicit features
        
        Parameters:
        - X: Feature matrix (n_samples, n_features)
        - roi_labels: ROI labels for each sample
        
        Returns:
        - X_with_coords: Feature matrix with spatial coordinates appended
        """
        n_samples = X.shape[0]
        
        # Create spatial feature arrays
        coord_x = np.zeros(n_samples)
        coord_y = np.zeros(n_samples)
        
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            row, col = self.coords[roi_idx]
            
            # Normalize coordinates to [0, 1] range
            coord_x[roi_mask] = col / (self.grid_size - 1)
            coord_y[roi_mask] = row / (self.grid_size - 1)
        
        # Append spatial coordinates as features
        X_with_coords = np.column_stack([X, coord_x, coord_y])
        
        return X_with_coords
    
    def generate_dataset_a_smooth_gradient(self, noise_level=0.25, normalize_data=False, global_boundaries=None):
        """
        Dataset A: Smooth gradient with spatial coordinates
        """
        print("Generating GeoSpatial Dataset A: Smooth Gradient + Coordinates")
        
        X_all, y_all, roi_labels = [], [], []
        roi_feature_weights = {}
        
        for roi_idx, (row, col) in enumerate(self.coords):
            # Same pattern generation as original
            r = row / (self.grid_size - 1)
            c = col / (self.grid_size - 1)
            diag_from_tl = (row + col) / (2 * (self.grid_size - 1))
            diag_from_br = ((self.grid_size - 1 - row) + (self.grid_size - 1 - col)) / (2 * (self.grid_size - 1))
            
            feature_means = np.zeros(self.n_features)
            feature_means[0] = 8.0 * (1 - diag_from_tl)
            feature_means[1] = 6.0 * (1 - diag_from_tl)
            feature_means[2] = 4.0 * (1 - diag_from_tl)
            feature_means[3] = 8.0 * (1 - diag_from_br)
            feature_means[4] = 6.0 * (1 - diag_from_br)
            feature_means[5] = 4.0 * (1 - diag_from_br)
            feature_means[6] = 6.0 * r
            feature_means[7] = 6.0 * c
            
            # Note: roi_feature_weights only stores non-spatial features
            roi_feature_weights[roi_idx] = feature_means.copy()
            
            X_roi = np.random.normal(loc=feature_means, scale=noise_level, 
                                    size=(self.n_samples_per_roi, self.n_features))
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)
        
        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)
        
        # Add spatial coordinates as features (now 10 features total)
        X_with_coords = self._add_spatial_features(X, roi_labels)
        
        # Generate target labels with custom approach for smooth gradient
        if global_boundaries is not None:
            # Ignore global boundaries and create local ones for better class balance
            beta = np.array([1.0, 0.8, 0.6, -0.5, -0.3, -0.1, 0.4, 0.4])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        else:
            beta = np.array([1.0, 0.8, 0.6, -0.5, -0.3, -0.1, 0.4, 0.4])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        
        return self._package_geospatial_dataset(X_with_coords, X, y, roi_labels, 
                                                "smooth_gradient", roi_feature_weights, 
                                                normalize_data)
    
    def generate_dataset_b_sharp_boundary(self, noise_level=0.05, normalize_data=False, global_boundaries=None):
        """
        Dataset B: Sharp boundary with spatial coordinates
        """
        print("Generating GeoSpatial Dataset B: Sharp Boundary + Coordinates")
        
        boundary_row = self.grid_size // 2
        X_all, roi_labels, roi_feature_weights = [], [], {}
        
        for roi_idx, (row, col) in enumerate(self.coords):
            base_features = np.array([4.0, 4.0, 3.5, 3.5, 3.0, 3.0, 2.5, 2.5])
            
            if row < boundary_row:
                feature_means = base_features * np.array([2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
            else:
                feature_means = base_features * np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
            
            roi_feature_weights[roi_idx] = feature_means.copy()
            
            X_roi = np.random.normal(loc=feature_means, scale=noise_level,
                                    size=(self.n_samples_per_roi, self.n_features))
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)
        
        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)
        
        # Add spatial coordinates as features (now 10 features total)
        X_with_coords = self._add_spatial_features(X, roi_labels)
        
        # Generate target labels - use custom beta for sharp boundary
        if global_boundaries is not None:
            # For sharp boundary, ignore global boundaries and create local ones
            beta = np.array([1.0, -1.0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        else:
            beta = np.array([1.0, -1.0, 0.9, -0.9, 0.8, -0.8, 0.7, -0.7])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        
        return self._package_geospatial_dataset(X_with_coords, X, y, roi_labels,
                                                "sharp_boundary", roi_feature_weights,
                                                normalize_data)
    
    def generate_dataset_c_clustered_hotspots(self, noise_level=0.05, n_hotspots=4, 
                                              normalize_data=False, global_boundaries=None):
        """
        Dataset C: Clustered hotspots with spatial coordinates
        """
        print("Generating GeoSpatial Dataset C: Clustered Hotspots + Coordinates")
        
        X_all, roi_labels, roi_feature_weights = [], [], {}
        
        # Hotspot centers
        # Define hotspot centers for each quadrant (matching SpatialPatternGenerator)
        g = self.grid_size
        hotspot_centers = [
            np.array([int(0.25 * (g - 1)), int(0.25 * (g - 1))]),  # Quadrant 1: top-left
            np.array([int(0.25 * (g - 1)), int(0.75 * (g - 1))]),  # Quadrant 2: top-right
            np.array([int(0.75 * (g - 1)), int(0.25 * (g - 1))]),  # Quadrant 3: bottom-left
            np.array([int(0.75 * (g - 1)), int(0.75 * (g - 1))]),  # Quadrant 4: bottom-right
        ]


        # Hotspot parameters
        radius = max(2.0, 0.3 * (g - 1))
        intensity = 8.0
        crosstalk = 0.10
        baseline = 0.30
        
        # Dominant feature pairs for each hotspot (same as SpatialPatternGenerator)
        dominant_features = [
            [0, 1],  # Hotspot 1: top-left
            [2, 3],  # Hotspot 2: top-right
            [4, 5],  # Hotspot 3: bottom-left
            [6, 7],  # Hotspot 4: bottom-right
        ]
        
        # Generate ROI data
        for roi_idx, (row, col) in enumerate(self.coords):
            coord = np.array([row, col], dtype=float)
            
            # Start with low baseline
            feature_means = np.full(self.n_features, baseline, dtype=float)
            
            # Add influence from each hotspot (Gaussian decay)
            for center, dom_features in zip(hotspot_centers, dominant_features):
                dist = np.linalg.norm(coord - center)
                
                if dist <= radius:
                    influence = intensity * np.exp(-(dist ** 2) / (radius ** 2))
                    
                    # Strong bump for dominant features
                    feature_means[dom_features] += influence
                    
                    # Mild spillover for other features
                    other_features = [k for k in range(self.n_features) if k not in dom_features]
                    feature_means[other_features] += crosstalk * influence
            
            roi_feature_weights[roi_idx] = feature_means.copy()
            
            X_roi = np.random.normal(loc=feature_means, scale=noise_level,
                                    size=(self.n_samples_per_roi, self.n_features))
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)
        
        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)

        X_with_coords = self._add_spatial_features(X, roi_labels)

        # Generate target labels with balanced classes for clustered hotspots
        if global_boundaries is not None:
            # Create local boundaries for better balance
            beta = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        else:
            beta = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
            weighted_sum = X @ beta
            p33, p67 = np.percentile(weighted_sum, [33, 67])
            y = np.digitize(weighted_sum, bins=[p33, p67])
        
        return self._package_geospatial_dataset(X_with_coords, X, y, roi_labels, 
                                                "clustered_hotspots", roi_feature_weights, 
                                                normalize_data)

    def generate_dataset_d_random_pattern(self, noise_level=0.05, normalize_data=False, global_boundaries=None):
        """
        Dataset D: Random pattern with spatial coordinates
        Total features: 10 (8 original + 2 spatial)
        """
        print("Generating GeoSpatial Dataset D: Random Pattern + Coordinates")
        
        X_all, roi_labels = [], []
        roi_feature_weights = {}
        
        for roi_idx, coord in enumerate(self.coords):
            # Random feature weights for this ROI
            feature_weights = np.random.uniform(0.1, 8.0, self.n_features)
            roi_feature_weights[roi_idx] = feature_weights.copy()
            
            # Generate samples
            X_roi = np.random.randn(self.n_samples_per_roi, self.n_features)
            
            # Scale by feature weights
            for i in range(self.n_features):
                X_roi[:, i] *= feature_weights[i]
            
            # Add noise
            X_roi += np.random.normal(0, noise_level, X_roi.shape)
            
            X_all.append(X_roi)
            roi_labels.extend([roi_idx] * self.n_samples_per_roi)
        
        X = np.vstack(X_all)
        roi_labels = np.array(roi_labels)
        
        # Add spatial coordinates
        X_with_coords = self._add_spatial_features(X, roi_labels)
        
        # Generate labels
        if global_boundaries is not None:
            global_p33, global_p67, beta = global_boundaries
            weighted_sum = X @ beta
            y = np.digitize(weighted_sum, bins=[global_p33, global_p67])
        else:
            beta = np.ones(self.n_features)
            weighted_sum = X @ beta
            y = np.digitize(weighted_sum, bins=np.percentile(weighted_sum, [33, 67]))
        
        return self._package_geospatial_dataset(X_with_coords, X, y, roi_labels,
                                                "random_pattern", roi_feature_weights,
                                                normalize_data)

    def select_consistent_samples(self, dataset_dict, random_state=42):
        """
        Select one representative sample per ROI for XAI analysis
        
        Parameters:
        - dataset_dict: Dataset dictionary from generate_dataset_* methods
        - random_state: Random seed for reproducibility
        
        Returns:
        - Dictionary with selected samples (100 samples total, 1 per ROI)
        """
        np.random.seed(random_state)
        
        X = dataset_dict['X']
        y = dataset_dict['y']
        roi_labels = dataset_dict['roi_labels']
        
        selected_indices = []
        
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            roi_indices = np.where(roi_mask)[0]
            
            if len(roi_indices) > 0:
                # Select random sample from this ROI
                selected_idx = np.random.choice(roi_indices)
                selected_indices.append(selected_idx)
        
        selected_indices = np.array(selected_indices)
        
        return {
            'indices': selected_indices,
            'X': X[selected_indices],
            'y': y[selected_indices],
            'roi_labels': np.arange(self.n_rois),  # One sample per ROI
            'dataset_name': dataset_dict['dataset_name'] + '_selected',
            'has_spatial_features': True,
            'spatial_feature_indices': dataset_dict['spatial_feature_indices']
        }

    def visualize_roi_grid(self, dataset_dict, feature_idx=0, show_values=True):
        """Visualize spatial pattern for a specific feature"""
        coords = dataset_dict['coords']
        X = dataset_dict.get('X_raw', dataset_dict['X'])
        roi_labels = dataset_dict['roi_labels']
        
        # Calculate mean feature value for each ROI
        roi_means = []
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            roi_mean = np.mean(X[roi_mask, feature_idx])
            roi_means.append(roi_mean)
        
        # Reshape to grid
        grid_values = np.array(roi_means).reshape(self.grid_size, self.grid_size)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(grid_values, cmap='RdYlBu_r', origin='lower')
        plt.colorbar(im, label=f'Mean Feature {feature_idx} Value', shrink=0.8)
        plt.title(f'{dataset_dict["dataset_name"].replace("_", " ").title()} - Feature {feature_idx}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Grid X Coordinate', fontsize=12)
        plt.ylabel('Grid Y Coordinate', fontsize=12)
        
        # Add grid lines
        for i in range(self.grid_size + 1):
            plt.axhline(i - 0.5, color='white', linewidth=1, alpha=0.7)
            plt.axvline(i - 0.5, color='white', linewidth=1, alpha=0.7)
        
        # Add value annotations
        if show_values:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    value = grid_values[i, j]
                    text_color = 'white' if value > np.mean(grid_values) else 'black'
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=text_color, fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _package_geospatial_dataset(self, X_with_coords, X_original, y, roi_labels, 
                                   dataset_name, roi_feature_weights, normalize_data):
        """
        Package dataset with spatial coordinates
        
        Parameters:
        - X_with_coords: Feature matrix with spatial coordinates
        - X_original: Original features without coordinates (for ground truth)
        - y: Labels
        - roi_labels: ROI labels
        - dataset_name: Dataset name
        - roi_feature_weights: Feature weights (non-spatial only)
        - normalize_data: Whether to normalize
        """
        return {
            'X': X_with_coords,  # 10 features: 8 original + 2 spatial
            'X_original': X_original,  # 8 original features only
            'y': y,
            'roi_labels': roi_labels,
            'coords': self.coords,
            'grid_size': self.grid_size,
            'n_features': self.n_features + 2,  # Total features including spatial
            'n_features_nonspatial': self.n_features,  # Original features only
            'dataset_name': dataset_name,
            'has_spatial_features': True,
            'spatial_feature_indices': [self.n_features, self.n_features + 1],  # Last two columns
            'roi_feature_weights': roi_feature_weights,  # Ground truth (non-spatial only)
            'scaler': None,
            'roi_scalers': None
        }


    def visualize_geospatial_features(self, dataset_dict, figsize=(16, 10)):
        """
        Visualize spatial coordinate features and sample non-spatial feature
        
        Parameters:
        - dataset_dict: GeoSpatial dataset dictionary
        - figsize: Figure size
        """
        X = dataset_dict['X']
        roi_labels = dataset_dict['roi_labels']
        spatial_indices = dataset_dict['spatial_feature_indices']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Coord X spatial distribution
        ax1 = axes[0, 0]
        coord_x_means = []
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            coord_x_means.append(np.mean(X[roi_mask, spatial_indices[0]]))
        
        coord_x_grid = np.array(coord_x_means).reshape(self.grid_size, self.grid_size)
        im1 = ax1.imshow(coord_x_grid, cmap='viridis', origin='lower')
        ax1.set_title('Spatial Feature: Coordinate X')
        ax1.set_xlabel('Grid Column')
        ax1.set_ylabel('Grid Row')
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        
        # Plot 2: Coord Y spatial distribution
        ax2 = axes[0, 1]
        coord_y_means = []
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            coord_y_means.append(np.mean(X[roi_mask, spatial_indices[1]]))
        
        coord_y_grid = np.array(coord_y_means).reshape(self.grid_size, self.grid_size)
        im2 = ax2.imshow(coord_y_grid, cmap='viridis', origin='lower')
        ax2.set_title('Spatial Feature: Coordinate Y')
        ax2.set_xlabel('Grid Column')
        ax2.set_ylabel('Grid Row')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # Plot 3: Sample non-spatial feature (Feature 0)
        ax3 = axes[1, 0]
        feature_0_means = []
        for roi_idx in range(self.n_rois):
            roi_mask = roi_labels == roi_idx
            feature_0_means.append(np.mean(X[roi_mask, 0]))
        
        feature_0_grid = np.array(feature_0_means).reshape(self.grid_size, self.grid_size)
        im3 = ax3.imshow(feature_0_grid, cmap='RdYlBu_r', origin='lower')
        ax3.set_title('Non-Spatial Feature 0 Pattern')
        ax3.set_xlabel('Grid Column')
        ax3.set_ylabel('Grid Row')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # Plot 4: Feature correlation heatmap (sample)
        ax4 = axes[1, 1]
        # Sample 1000 points for correlation calculation
        sample_size = min(1000, X.shape[0])
        sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[sample_indices]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_sample.T)
        
        im4 = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title('Feature Correlation Matrix\n(8 non-spatial + 2 spatial)')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Feature Index')
        
        # Add feature labels
        feature_labels = [f'F{i}' for i in range(self.n_features)] + ['X', 'Y']
        ax4.set_xticks(range(len(feature_labels)))
        ax4.set_yticks(range(len(feature_labels)))
        ax4.set_xticklabels(feature_labels, rotation=45)
        ax4.set_yticklabels(feature_labels)
        
        plt.colorbar(im4, ax=ax4, shrink=0.8)
        
        plt.suptitle(f'GeoSpatial Dataset Visualization: {dataset_dict["dataset_name"].replace("_", " ").title()}',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig
```

## PART 3: DATASET GENERATION WORKFLOW


```python
def generate_datasets_group_1(grid_size=10, n_features=8, n_samples_per_roi=100, 
                               use_global_boundaries=True, random_state=42):
    """
    Generate Group 1 datasets (8 features, raw data for SHAP+LIME comparison)
    
    Parameters:
    - grid_size: Size of spatial grid
    - n_features: Number of features (should be 8 for Group 1)
    - n_samples_per_roi: Samples per ROI
    - use_global_boundaries: Whether to use global class boundaries
    - random_state: Random seed
    
    Returns:
    - Dictionary of datasets
    """
    print("="*70)
    print("GROUP 1: Generating 8-variable datasets (raw data for SHAP+LIME)")
    print("="*70)
    
    np.random.seed(random_state)
    
    generator = SpatialPatternGenerator(
        grid_size=grid_size,
        n_features=n_features,
        n_samples_per_roi=n_samples_per_roi
    )
    
    # Calculate global boundaries if requested
    global_boundaries = None
    if use_global_boundaries:
        print("Calculating global class boundaries...")
        beta = np.array([1.0, 0.8, 0.6, 1.0, 0.8, 0.6, 0.7, 0.7])
        all_weighted_sums = []
        
        # Generate small samples from each pattern for boundary calculation
        temp_gen = SpatialPatternGenerator(grid_size=grid_size, n_features=n_features, n_samples_per_roi=10)
        temp_datasets = [
            temp_gen.generate_dataset_a_smooth_gradient(normalize_data=False),
            temp_gen.generate_dataset_b_sharp_boundary(normalize_data=False),
            temp_gen.generate_dataset_c_clustered_hotspots(normalize_data=False),
            temp_gen.generate_dataset_d_random_pattern(normalize_data=False)
        ]
        
        # Collect weighted sums
        for temp_dataset in temp_datasets:
            weighted_sum = temp_dataset['X'] @ beta
            all_weighted_sums.extend(weighted_sum)
        
        # Calculate global thresholds
        global_p33, global_p67 = np.percentile(all_weighted_sums, [33, 67])
        global_boundaries = (global_p33, global_p67, beta)
        print(f"Global boundaries: [{global_p33:.3f}, {global_p67:.3f}]")
    
    # Generate all datasets WITHOUT normalization (raw data for SHAP)
    print("\nGenerating datasets with spatial patterns...")
    dataset_a = generator.generate_dataset_a_smooth_gradient(normalize_data=False, global_boundaries=global_boundaries)
    dataset_b = generator.generate_dataset_b_sharp_boundary(normalize_data=False, global_boundaries=global_boundaries)
    dataset_c = generator.generate_dataset_c_clustered_hotspots(normalize_data=False, global_boundaries=global_boundaries)
    dataset_d = generator.generate_dataset_d_random_pattern(normalize_data=False, global_boundaries=global_boundaries)
    
    datasets = {
        'smooth_gradient': dataset_a,
        'sharp_boundary': dataset_b,
        'clustered_hotspots': dataset_c,
        'random_pattern': dataset_d
    }
    
    # Print summary
    print(f"\nDataset shapes (all using raw data):")
    for name, dataset in datasets.items():
        X_is_raw = np.array_equal(dataset['X'], dataset['X_raw'])
        print(f"  {name}: X={dataset['X'].shape}, y={dataset['y'].shape}, X_is_raw={X_is_raw}")
    
    print("\n" + "="*70)
    print("GROUP 1 datasets ready for MLP training and SHAP+LIME analysis")
    print("="*70)
    
    return datasets


def generate_datasets_group_2(grid_size=10, n_features=8, n_samples_per_roi=100,
                               use_global_boundaries=True, random_state=42):
    """
    Generate Group 2 datasets (10 features = 8 + 2 spatial coords, for SHAP+GeoSHAP comparison)
    
    Parameters:
    - grid_size: Size of spatial grid
    - n_features: Number of non-spatial features (8, spatial coords added automatically)
    - n_samples_per_roi: Samples per ROI
    - use_global_boundaries: Whether to use global class boundaries
    - random_state: Random seed
    
    Returns:
    - Dictionary of datasets (each with 10 features total)
    """
    print("="*70)
    print("GROUP 2: Generating 10-variable datasets (8 features + 2 coords for SHAP+GeoSHAP)")
    print("="*70)
    
    np.random.seed(random_state)
    
    # Use GeoSpatialPatternGenerator (adds spatial coordinates automatically)
    generator = GeoSpatialPatternGenerator(
        grid_size=grid_size,
        n_features=n_features,  # 8 non-spatial features
        n_samples_per_roi=n_samples_per_roi
    )
    
    # Calculate global boundaries if requested
    global_boundaries = None
    if use_global_boundaries:
        print("Calculating global class boundaries...")
        beta = np.array([1.0, 0.8, 0.6, 1.0, 0.8, 0.6, 0.7, 0.7])  # 8 features
        all_weighted_sums = []
        
        # Generate small samples from each pattern for boundary calculation
        temp_gen = GeoSpatialPatternGenerator(grid_size=grid_size, n_features=n_features, n_samples_per_roi=10)
        temp_datasets = [
            temp_gen.generate_dataset_a_smooth_gradient(normalize_data=False),
            temp_gen.generate_dataset_b_sharp_boundary(normalize_data=False),
            temp_gen.generate_dataset_c_clustered_hotspots(normalize_data=False),
            temp_gen.generate_dataset_d_random_pattern(normalize_data=False)
        ]
        
        # Collect weighted sums (using only non-spatial features)
        for temp_dataset in temp_datasets:
            X_original = temp_dataset['X_original']  # Non-spatial features only
            weighted_sum = X_original @ beta
            all_weighted_sums.extend(weighted_sum)
        
        # Calculate global thresholds
        global_p33, global_p67 = np.percentile(all_weighted_sums, [33, 67])
        global_boundaries = (global_p33, global_p67, beta)
        print(f"Global boundaries: [{global_p33:.3f}, {global_p67:.3f}]")
    
    # Generate all datasets
    print("\nGenerating datasets with spatial patterns...")
    dataset_a = generator.generate_dataset_a_smooth_gradient(normalize_data=False, global_boundaries=global_boundaries)
    dataset_b = generator.generate_dataset_b_sharp_boundary(normalize_data=False, global_boundaries=global_boundaries)
    dataset_c = generator.generate_dataset_c_clustered_hotspots(normalize_data=False, global_boundaries=global_boundaries)
    dataset_d = generator.generate_dataset_d_random_pattern(normalize_data=False, global_boundaries=global_boundaries)
    
    datasets = {
        'smooth_gradient': dataset_a,
        'sharp_boundary': dataset_b,
        'clustered_hotspots': dataset_c,
        'random_pattern': dataset_d
    }
    
    # Print summary
    print(f"\nDataset shapes (10 features = 8 original + 2 spatial):")
    for name, dataset in datasets.items():
        print(f"  {name}: X={dataset['X'].shape}, y={dataset['y'].shape}, "
              f"spatial_indices={dataset['spatial_feature_indices']}")
    
    print("\n" + "="*70)
    print("GROUP 2 datasets ready for MLP training and SHAP+GeoSHAP analysis")
    print("="*70)
    
    return datasets


def visualize_all_datasets(generator, datasets, feature_idx=0):
    """
    Visualize spatial patterns for all datasets
    
    Parameters:
    - generator: SpatialPatternGenerator instance
    - datasets: Dictionary of datasets
    - feature_idx: Feature index to visualize
    """
    print(f"\nVisualizing spatial patterns for feature {feature_idx}...")
    
    for name, dataset in datasets.items():
        generator.visualize_roi_grid(dataset, feature_idx=feature_idx)


def analyze_spatial_statistics(generator, datasets, feature_indices=None):
    """
    Calculate and visualize spatial autocorrelation statistics for all datasets
    
    Parameters:
    - generator: SpatialPatternGenerator instance
    - datasets: Dictionary of datasets
    - feature_indices: List of feature indices to analyze (None = first 4 features)
    """
    if feature_indices is None:
        feature_indices = [0, 1, 2, 3]
    
    print("\n" + "="*70)
    print("SPATIAL AUTOCORRELATION ANALYSIS")
    print("="*70)
    
    all_statistics = {}
    for name, dataset in datasets.items():
        all_statistics[name] = generator.calculate_spatial_statistics(
            dataset, 
            feature_indices=feature_indices
        )
    
    return all_statistics


def select_samples_for_xai(generator, datasets, random_state=42):
    """
    Select one representative sample per ROI from each dataset for XAI analysis
    
    Parameters:
    - generator: SpatialPatternGenerator or GeoSpatialPatternGenerator instance
    - datasets: Dictionary of datasets
    - random_state: Random seed
    
    Returns:
    - Dictionary of selected samples for each dataset
    """
    print("\nSelecting representative samples for XAI analysis...")
    
    selected_datasets = {}
    for name, dataset in datasets.items():
        selected = generator.select_consistent_samples(dataset, random_state=random_state)
        selected_datasets[name] = selected
        print(f"  {name}: Selected {len(selected['indices'])} samples (1 per ROI)")
    
    return selected_datasets
```

## Part 4: Implementation


```python
if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPATIAL XAI DATASET GENERATION PIPELINE")
    print("="*70)
    
    # ========================================================================
    # GROUP 1: 8-variable datasets for SHAP + LIME comparison
    # ========================================================================
    
    print("\n\n### GENERATING GROUP 1 DATASETS ###\n")
    
    datasets_group1 = generate_datasets_group_1(
        grid_size=10,
        n_features=8,
        n_samples_per_roi=100,
        use_global_boundaries=True,
        random_state=42
    )
    
    # Create generator instance for visualization/analysis
    generator1 = SpatialPatternGenerator(grid_size=10, n_features=8, n_samples_per_roi=100)

      
    # Visualize patterns
    print("\n### VISUALIZING GROUP 1 SPATIAL PATTERNS ###")
    visualize_all_datasets(generator1, datasets_group1, feature_idx=0)

    # Visualize different features to see all hotspots
    generator1.visualize_roi_grid(datasets_group1['clustered_hotspots'], feature_idx=0)  # Top-left
    generator1.visualize_roi_grid(datasets_group1['clustered_hotspots'], feature_idx=2)  # Top-right
    generator1.visualize_roi_grid(datasets_group1['clustered_hotspots'], feature_idx=4)  # Bottom-left
    generator1.visualize_roi_grid(datasets_group1['clustered_hotspots'], feature_idx=6)  # Bottom-right
    
    # Calculate spatial statistics
    print("\n### CALCULATING GROUP 1 SPATIAL STATISTICS ###")
    stats_group1 = analyze_spatial_statistics(
        generator1, 
        datasets_group1, 
        feature_indices=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    
    # Select samples for XAI
    print("\n### SELECTING GROUP 1 SAMPLES FOR XAI ###")
    selected_group1 = select_samples_for_xai(generator1, datasets_group1, random_state=42)
    
    print("\n" + "="*70)
    print("GROUP 1 COMPLETE - Ready for SHAP + LIME analysis")
    print("="*70)
    
    # ========================================================================
    # GROUP 2: 10-variable datasets (8 + 2 spatial coords) for SHAP + GeoSHAP
    # ========================================================================
    
    print("\n\n### GENERATING GROUP 2 DATASETS ###\n")
    
    datasets_group2 = generate_datasets_group_2(
        grid_size=10,
        n_features=8,  # Non-spatial features (spatial coords added automatically)
        n_samples_per_roi=100,
        use_global_boundaries=True,
        random_state=42
    )

    # Create generator instance for Group 2
    generator2 = GeoSpatialPatternGenerator(grid_size=10, n_features=8, n_samples_per_roi=100)
    
    # Visualize patterns for Group 2
    print("\n### VISUALIZING GROUP 2 SPATIAL PATTERNS ###")
    visualize_all_datasets(generator2, datasets_group2, feature_idx=0)

    # Visualize different features to see all hotspots
    generator2.visualize_roi_grid(datasets_group2['clustered_hotspots'], feature_idx=0)  # Top-left
    generator2.visualize_roi_grid(datasets_group2['clustered_hotspots'], feature_idx=2)  # Top-right
    generator2.visualize_roi_grid(datasets_group2['clustered_hotspots'], feature_idx=4)  # Bottom-left
    generator2.visualize_roi_grid(datasets_group2['clustered_hotspots'], feature_idx=6)  # Bottom-right

    # Calculate spatial statistics
    print("\n### CALCULATING GROUP 2 SPATIAL STATISTICS ###")
    stats_group2 = analyze_spatial_statistics(
        generator2, 
        datasets_group2, 
        feature_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    
    # Select samples for XAI
    print("\n### SELECTING GROUP 2 SAMPLES FOR XAI ###")
    selected_group2 = select_samples_for_xai(generator2, datasets_group2, random_state=42)
    
    print("\n" + "="*70)
    print("GROUP 2 COMPLETE - Ready for SHAP + GeoSHAP analysis")
    print("="*70)
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n\n" + "="*70)
    print("DATASET GENERATION SUMMARY")
    print("="*70)
    
    print("\nGROUP 1 (SHAP + LIME):")
    print(f"  Datasets: {list(datasets_group1.keys())}")
    print(f"  Features: 8 (raw data, no spatial coordinates)")
    print(f"  Samples per dataset: {datasets_group1['smooth_gradient']['X'].shape[0]}")
    print(f"  Selected samples per dataset: {selected_group1['smooth_gradient']['X'].shape[0]}")
    
    print("\nGROUP 2 (SHAP + GeoSHAP):")
    print(f"  Datasets: {list(datasets_group2.keys())}")
    print(f"  Features: 10 (8 original + 2 spatial coordinates)")
    print(f"  Spatial feature indices: {datasets_group2['smooth_gradient']['spatial_feature_indices']}")
    print(f"  Samples per dataset: {datasets_group2['smooth_gradient']['X'].shape[0]}")
    print(f"  Selected samples per dataset: {selected_group2['smooth_gradient']['X'].shape[0]}")
```

    
    ======================================================================
    SPATIAL XAI DATASET GENERATION PIPELINE
    ======================================================================
    
    
    ### GENERATING GROUP 1 DATASETS ###
    
    ======================================================================
    GROUP 1: Generating 8-variable datasets (raw data for SHAP+LIME)
    ======================================================================
    Calculating global class boundaries...
    Generating Dataset A: Smooth Spatial Gradients
    Generating Dataset B: Sharp Boundary Effect
    Generating Dataset C: Clustered Hotspots Pattern (4 quadrants)
    Generating Dataset D: Random Pattern (Negative Control)
    Global boundaries: [10.881, 20.581]
    
    Generating datasets with spatial patterns...
    Generating Dataset A: Smooth Spatial Gradients
    Generating Dataset B: Sharp Boundary Effect
    Generating Dataset C: Clustered Hotspots Pattern (4 quadrants)
    Generating Dataset D: Random Pattern (Negative Control)
    
    Dataset shapes (all using raw data):
      smooth_gradient: X=(10000, 8), y=(10000,), X_is_raw=True
      sharp_boundary: X=(10000, 8), y=(10000,), X_is_raw=True
      clustered_hotspots: X=(10000, 8), y=(10000,), X_is_raw=True
      random_pattern: X=(10000, 8), y=(10000,), X_is_raw=True
    
    ======================================================================
    GROUP 1 datasets ready for MLP training and SHAP+LIME analysis
    ======================================================================
    
    ### VISUALIZING GROUP 1 SPATIAL PATTERNS ###
    
    Visualizing spatial patterns for feature 0...
    


    
![png](output_14_1.png)
    



    
![png](output_14_2.png)
    



    
![png](output_14_3.png)
    



    
![png](output_14_4.png)
    



    
![png](output_14_5.png)
    



    
![png](output_14_6.png)
    



    
![png](output_14_7.png)
    



    
![png](output_14_8.png)
    


    
    ### CALCULATING GROUP 1 SPATIAL STATISTICS ###
    
    ======================================================================
    SPATIAL AUTOCORRELATION ANALYSIS
    ======================================================================
    
    Calculating spatial autocorrelation for smooth_gradient...
      Feature 0: Moran's I = 0.932 ***, Geary's C = 0.044
      Feature 1: Moran's I = 0.932 ***, Geary's C = 0.044
      Feature 2: Moran's I = 0.933 ***, Geary's C = 0.044
      Feature 3: Moran's I = 0.932 ***, Geary's C = 0.044
    
    Calculating spatial autocorrelation for sharp_boundary...
      Feature 0: Moran's I = 0.848 ***, Geary's C = 0.150
      Feature 1: Moran's I = 0.848 ***, Geary's C = 0.150
      Feature 2: Moran's I = 0.848 ***, Geary's C = 0.150
      Feature 3: Moran's I = 0.848 ***, Geary's C = 0.151
    
    Calculating spatial autocorrelation for clustered_hotspots...
      Feature 0: Moran's I = 0.779 ns, Geary's C = 0.246
      Feature 1: Moran's I = 0.778 ns, Geary's C = 0.246
      Feature 2: Moran's I = 0.730 ns, Geary's C = 0.282
      Feature 3: Moran's I = 0.730 ns, Geary's C = 0.282
    
    Calculating spatial autocorrelation for random_pattern...
      Feature 0: Moran's I = 0.013 ns, Geary's C = 0.973
      Feature 1: Moran's I = 0.017 ns, Geary's C = 0.962
      Feature 2: Moran's I = -0.007 ns, Geary's C = 1.025
      Feature 3: Moran's I = -0.061 ns, Geary's C = 1.037
    
    ### SELECTING GROUP 1 SAMPLES FOR XAI ###
    
    Selecting representative samples for XAI analysis...
      smooth_gradient: Selected 100 samples (1 per ROI)
      sharp_boundary: Selected 100 samples (1 per ROI)
      clustered_hotspots: Selected 100 samples (1 per ROI)
      random_pattern: Selected 100 samples (1 per ROI)
    
    ======================================================================
    GROUP 1 COMPLETE - Ready for SHAP + LIME analysis
    ======================================================================
    
    
    ### GENERATING GROUP 2 DATASETS ###
    
    ======================================================================
    GROUP 2: Generating 10-variable datasets (8 features + 2 coords for SHAP+GeoSHAP)
    ======================================================================
    Calculating global class boundaries...
    Generating GeoSpatial Dataset A: Smooth Gradient + Coordinates
    Generating GeoSpatial Dataset B: Sharp Boundary + Coordinates
    Generating GeoSpatial Dataset C: Clustered Hotspots + Coordinates
    Generating GeoSpatial Dataset D: Random Pattern + Coordinates
    Global boundaries: [10.881, 20.581]
    
    Generating datasets with spatial patterns...
    Generating GeoSpatial Dataset A: Smooth Gradient + Coordinates
    Generating GeoSpatial Dataset B: Sharp Boundary + Coordinates
    Generating GeoSpatial Dataset C: Clustered Hotspots + Coordinates
    Generating GeoSpatial Dataset D: Random Pattern + Coordinates
    
    Dataset shapes (10 features = 8 original + 2 spatial):
      smooth_gradient: X=(10000, 10), y=(10000,), spatial_indices=[8, 9]
      sharp_boundary: X=(10000, 10), y=(10000,), spatial_indices=[8, 9]
      clustered_hotspots: X=(10000, 10), y=(10000,), spatial_indices=[8, 9]
      random_pattern: X=(10000, 10), y=(10000,), spatial_indices=[8, 9]
    
    ======================================================================
    GROUP 2 datasets ready for MLP training and SHAP+GeoSHAP analysis
    ======================================================================
    
    ### VISUALIZING GROUP 2 SPATIAL PATTERNS ###
    
    Visualizing spatial patterns for feature 0...
    


    
![png](output_14_10.png)
    



    
![png](output_14_11.png)
    



    
![png](output_14_12.png)
    



    
![png](output_14_13.png)
    



    
![png](output_14_14.png)
    



    
![png](output_14_15.png)
    



    
![png](output_14_16.png)
    



    
![png](output_14_17.png)
    


    
    ### SELECTING GROUP 2 SAMPLES FOR XAI ###
    
    Selecting representative samples for XAI analysis...
      smooth_gradient: Selected 100 samples (1 per ROI)
      sharp_boundary: Selected 100 samples (1 per ROI)
      clustered_hotspots: Selected 100 samples (1 per ROI)
      random_pattern: Selected 100 samples (1 per ROI)
    
    ======================================================================
    GROUP 2 COMPLETE - Ready for SHAP + GeoSHAP analysis
    ======================================================================
    
    
    ======================================================================
    DATASET GENERATION SUMMARY
    ======================================================================
    
    GROUP 1 (SHAP + LIME):
      Datasets: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
      Features: 8 (raw data, no spatial coordinates)
      Samples per dataset: 10000
      Selected samples per dataset: 100
    
    GROUP 2 (SHAP + GeoSHAP):
      Datasets: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
      Features: 10 (8 original + 2 spatial coordinates)
      Spatial feature indices: [8, 9]
      Samples per dataset: 10000
      Selected samples per dataset: 100
    

## Create MLPs with one hidden layer


```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input  
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt
import types
import os
```


```python
class SimpleMLPModels:
    def __init__(self, random_state=42):
        """
        Simple MLP model trainer for XAI framework testing

        Parameters:
        - random_state: For reproducible results
        """
        self.random_state = random_state
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        self.models = {}
        self.histories = {}
        self.test_data = {}
        self._logits_models = {}  # Store logits versions

    def create_simple_model(self, input_dim, n_classes=3, hidden_units=64):
        """
        Create a simple neural network with one hidden layer

        Parameters:
        - input_dim: Number of input features
        - n_classes: Number of output classes
        - hidden_units: Number of neurons in hidden layer
        """
        model = Sequential([
            Input(shape=(input_dim,), name='input'),
            Dense(hidden_units, activation='relu', name='hidden_layer'),
            Dropout(0.2, name='dropout'),
            Dense(n_classes, activation='softmax', name='output_layer'),
        ], name='mlp_classifier')

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_model(self, dataset_dict, test_size=0.2, epochs=50, batch_size=32, verbose=1, callbacks=None):
        """
        Train a model on the given dataset

        Parameters:
        - dataset_dict: Dictionary containing X, y, and metadata
        - test_size: Proportion of data for testing
        - epochs: Number of training epochs
        - batch_size: Training batch size
        - verbose: Verbosity level
        """
        dataset_name = dataset_dict['dataset_name']
        print(f"\nTraining model for {dataset_name}...")

        X = dataset_dict['X']
        y = dataset_dict['y']

        # Convert to categorical (one-hot encoding)
        y_categorical = to_categorical(y, num_classes=3)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )

        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

        # Create model
        model = self.create_simple_model(
            input_dim=X.shape[1],
            n_classes=3
        )

        print(f"\nModel architecture:")
        model.summary()

        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            shuffle=True,
            callbacks=callbacks or []
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        # Predictions for classification report
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        print(f"Class distribution in test set: {np.bincount(y_test_classes)}")
        print(f"Class distribution in predictions: {np.bincount(y_pred_classes)}")

        print(f"\nClassification Report:")
        print(classification_report(y_test_classes, y_pred_classes, zero_division=0))

        # Store results
        self.models[dataset_name] = model
        self.histories[dataset_name] = history
        self.test_data[dataset_name] = {
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': test_accuracy
        }

        return model, history

    def train_all_datasets(self, dataset_a, dataset_b, dataset_c, dataset_d, **kwargs):
        """
        Train models on all datasets

        Parameters:
        - dataset_a, dataset_b, dataset_c, dataset_d: Dataset dictionaries
        - **kwargs: Additional arguments for train_model
        """
        datasets = [dataset_a, dataset_b, dataset_c, dataset_d]

        print("="*60)
        print("TRAINING MODELS FOR ALL DATASETS")
        print("="*60)

        for dataset in datasets:
            self.train_model(dataset, **kwargs)
            print("\n" + "-"*50)

        # Summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)

        for name, test_data in self.test_data.items():
            print(f"{name}: Accuracy = {test_data['accuracy']:.4f}")

    def create_logits_model(self, dataset_name):
        """
        Create a logits version of an existing trained model
        
        Parameters:
        - dataset_name: Name of the dataset/model to convert
        
        Returns:
        - logits_model: Model that outputs logits (pre-softmax) instead of probabilities
        """
        if dataset_name not in self.models:
            raise ValueError(f"No trained model found for dataset: {dataset_name}")
        
        # Check if logits model already exists
        if dataset_name in self._logits_models:
            return self._logits_models[dataset_name]
        
        original_model = self.models[dataset_name]
        
        print(f"Converting {dataset_name} model to logits output...")
        
        # Ensure the model is built
        if not original_model.built:
            print("Model not built yet, building with sample input...")
            sample_input = np.zeros((1, original_model.layers[0].input_spec.shape[-1]))
            _ = original_model(sample_input)
        
        # Create new model by cloning architecture and changing final activation
        model_config = original_model.get_config()
        
        # Modify the final layer activation to linear
        model_config['layers'][-1]['config']['activation'] = 'linear'
        model_config['layers'][-1]['config']['name'] = f"{model_config['layers'][-1]['config']['name']}_logits"
        
        # Create new model from modified config
        logits_model = tf.keras.Sequential.from_config(model_config)
        logits_model._name = f"{original_model.name}_logits"
        
        # Build the new model with the same input shape
        logits_model.build(input_shape=original_model.input_shape)
        
        # Copy all weights from original model
        logits_model.set_weights(original_model.get_weights())
        
        # Store the logits model
        self._logits_models[dataset_name] = logits_model
        
        print(f"✓ Logits model created: softmax → linear")
        print(f"  Original activation: {original_model.layers[-1].activation.__name__}")
        print(f"  New activation: {logits_model.layers[-1].activation.__name__}")
        
        return logits_model

    def get_logits_prediction(self, dataset_name, X, verbose=0):
        """
        Get logits predictions for a dataset using the corresponding trained model
        
        Parameters:
        - dataset_name: Name of the dataset/model to use
        - X: Input data
        - verbose: Verbosity level for prediction
        
        Returns:
        - logits: Raw logits (pre-softmax) predictions
        """
        # Create logits model if it doesn't exist
        if dataset_name not in self._logits_models:
            self.create_logits_model(dataset_name)
        
        logits_model = self._logits_models[dataset_name]
        return logits_model.predict(X, verbose=verbose)

    def get_probabilities_prediction(self, dataset_name, X, verbose=0):
        """
        Get probability predictions using the original softmax model
        
        Parameters:
        - dataset_name: Name of the dataset/model to use  
        - X: Input data
        - verbose: Verbosity level for prediction
        
        Returns:
        - probabilities: Softmax probabilities
        """
        if dataset_name not in self.models:
            raise ValueError(f"No trained model found for dataset: {dataset_name}")
        
        return self.models[dataset_name].predict(X, verbose=verbose)

    def get_feature_importance_gradients(self, dataset_name, X, target_class=None):
        """
        Calculate gradient-based feature importance (gradient × input)
        
        Parameters:
        - dataset_name: Name of the dataset/model to use
        - X: Input data
        - target_class: Target class for gradient calculation (None = predicted class)
        
        Returns:
        - gradients: Raw gradients
        - importances: Gradient × input feature importances
        """
        # Ensure logits model exists
        if dataset_name not in self._logits_models:
            self.create_logits_model(dataset_name)
        
        logits_model = self._logits_models[dataset_name]
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            logits = logits_model(X_tensor)
            
            if target_class is None:
                target_class = tf.argmax(logits, axis=1)
            
            target_logits = tf.gather(logits, target_class, axis=1, batch_dims=1)
        
        gradients = tape.gradient(target_logits, X_tensor)
        
        # Feature importance = gradient × input
        importances = (gradients * X_tensor).numpy()
        
        return gradients.numpy(), importances

    def compare_logits_vs_probabilities(self, dataset_name, X_sample, verbose=1):
        """
        Compare logits vs probabilities output for verification
        
        Parameters:
        - dataset_name: Name of dataset/model
        - X_sample: Sample data to test (should be small for readability)
        - verbose: Print details
        
        Returns:
        - comparison_dict: Dictionary with logits, probabilities, and verification
        """
        logits = self.get_logits_prediction(dataset_name, X_sample, verbose=0)
        probabilities = self.get_probabilities_prediction(dataset_name, X_sample, verbose=0)
        
        # Verify: softmax(logits) should equal probabilities
        probabilities_from_logits = softmax(logits, axis=1)
        
        max_diff = np.max(np.abs(probabilities - probabilities_from_logits))
        
        if verbose:
            print(f"Logits vs Probabilities Comparison for {dataset_name}:")
            print(f"Sample shape: {X_sample.shape}")
            print(f"Logits shape: {logits.shape}")
            print(f"Probabilities shape: {probabilities.shape}")
            print(f"Max difference between P(softmax(logits)) and P(original): {max_diff:.2e}")
            
            print(f"\nSample logits[0]: {logits[0]}")
            print(f"Sample probabilities[0]: {probabilities[0]}")
            print(f"Sample softmax(logits)[0]: {probabilities_from_logits[0]}")
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'probabilities_from_logits': probabilities_from_logits,
            'max_difference': max_diff,
            'verification_passed': max_diff < 1e-6
        }

    def plot_training_history(self, dataset_name=None):
        """
        Plot training history for one or all models

        Parameters:
        - dataset_name: Specific dataset to plot, or None for all
        """
        if dataset_name:
            histories_to_plot = {dataset_name: self.histories[dataset_name]}
        else:
            histories_to_plot = self.histories

        n_plots = len(histories_to_plot)
        fig, axes = plt.subplots(n_plots, 2, figsize=(12, 4*n_plots))

        if n_plots == 1:
            axes = np.array(axes).reshape(1, -1)

        for idx, (name, history) in enumerate(histories_to_plot.items()):
            # Plot accuracy
            axes[idx, 0].plot(history.history['accuracy'], label='Training Accuracy')
            axes[idx, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
            axes[idx, 0].set_title(f'{name} - Model Accuracy')
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Accuracy')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)

            # Plot loss
            axes[idx, 1].plot(history.history['loss'], label='Training Loss')
            axes[idx, 1].plot(history.history['val_loss'], label='Validation Loss')
            axes[idx, 1].set_title(f'{name} - Model Loss')
            axes[idx, 1].set_xlabel('Epoch')
            axes[idx, 1].set_ylabel('Loss')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def get_model(self, dataset_name):
        """Get trained model for specific dataset"""
        return self.models.get(dataset_name)

    def get_logits_model(self, dataset_name):
        """Get logits model for specific dataset"""
        if dataset_name not in self._logits_models:
            self.create_logits_model(dataset_name)
        return self._logits_models.get(dataset_name)

    def get_test_data(self, dataset_name):
        """Get test data for specific dataset"""
        return self.test_data.get(dataset_name)

    def predict_by_roi(self, dataset_dict, roi_id):
        """
        Get predictions for all samples from a specific ROI

        Parameters:
        - dataset_dict: Dataset dictionary
        - roi_id: ROI identifier (0 to n_rois-1)
        """
        dataset_name = dataset_dict['dataset_name']
        model = self.models[dataset_name]

        # Get all samples for this ROI
        roi_mask = dataset_dict['roi_labels'] == roi_id
        X_roi = dataset_dict['X'][roi_mask]

        # Get predictions
        predictions = model.predict(X_roi, verbose=0)

        return {
            'predictions': predictions,
            'predicted_classes': np.argmax(predictions, axis=1),
            'confidence': np.max(predictions, axis=1),
            'n_samples': len(X_roi)
        }

    def save_models(self, save_path="./models/"):
        """Save all trained models"""
        import os
        os.makedirs(save_path, exist_ok=True)
    
        for name, model in self.models.items():
            model_path = os.path.join(save_path, f"{name}_model.keras")
            model.save(model_path)
            print(f"Model saved: {model_path}")

    def test_logits_functionality(self, dataset_dict):
        """
        Test the logits functionality with a dataset
        
        Parameters:
        - dataset_dict: Dataset dictionary to test with
        """
        dataset_name = dataset_dict['dataset_name']
        print(f"Testing Logits Functionality for {dataset_name}")
        print("=" * 50)
        
        # Test with a small sample
        X_test = dataset_dict['X'][:5]  # First 5 samples
        
        # Test the comparison function
        comparison = self.compare_logits_vs_probabilities(dataset_name, X_test, verbose=1)
        
        print(f"\nAdditivity test with logits:")
        logits = self.get_logits_prediction(dataset_name, X_test)
        
        # Check that logits sum appropriately (they don't need to sum to 1 like probabilities)
        print(f"Logits sum per sample: {np.sum(logits, axis=1)}")
        print(f"Probabilities sum per sample: {np.sum(comparison['probabilities'], axis=1)}")
        
        # Test gradient functionality
        print(f"\nTesting gradient functionality:")
        gradients, importances = self.get_feature_importance_gradients(dataset_name, X_test)
        print(f"Gradients shape: {gradients.shape}")
        print(f"Importances shape: {importances.shape}")
        print(f"Sample importance[0]: {importances[0]}")
        
        return comparison
```


```python
def train_models_workflow(datasets_dict, group_name="Group 1", save_prefix=""):
    """
    Train models on provided datasets
    
    Parameters:
    - datasets_dict: Dictionary with keys 'smooth_gradient', 'sharp_boundary', etc.
    - group_name: Name for printing (e.g., "Group 1", "Group 2")
    - save_prefix: Prefix for saved model files (e.g., "group1_", "group2_")
    """
    print(f"\n{'='*70}")
    print(f"TRAINING {group_name} MODELS")
    print(f"{'='*70}\n")
    
    # Initialize trainer
    trainer = SimpleMLPModels(random_state=42)
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    
    # Extract datasets
    dataset_a = datasets_dict['smooth_gradient']
    dataset_b = datasets_dict['sharp_boundary']
    dataset_c = datasets_dict['clustered_hotspots']
    dataset_d = datasets_dict['random_pattern']
    
    # Train all models
    trainer.train_all_datasets(
        dataset_a, dataset_b, dataset_c, dataset_d,
        epochs=50,
        batch_size=32,
        verbose=0,  # Less verbose for cleaner output
        callbacks=callbacks
    )
    
    # Plot training history
    trainer.plot_training_history()
    
    # Test logits functionality
    print(f"\n{'='*70}")
    print(f"TESTING LOGITS FUNCTIONALITY")
    print(f"{'='*70}\n")
    trainer.test_logits_functionality(dataset_a)
    
    # Save models with prefix
    if save_prefix:
        trainer.save_models(save_path=f"./models/{save_prefix}")
    
    print(f"\n{group_name} training completed. Models trained: {list(trainer.models.keys())}")
    
    return trainer


```


```python
# Check class distributions
for name, dataset in datasets_group1.items():
    print(f"{name}: Class distribution = {np.bincount(dataset['y'])}")

# Check class distributions
for name, dataset in datasets_group2.items():
    print(f"{name}: Class distribution = {np.bincount(dataset['y'])}")
```

    smooth_gradient: Class distribution = [3300 3400 3300]
    sharp_boundary: Class distribution = [3300 3400 3300]
    clustered_hotspots: Class distribution = [3300 3400 3300]
    random_pattern: Class distribution = [8589 1140  271]
    smooth_gradient: Class distribution = [3300 3400 3300]
    sharp_boundary: Class distribution = [3300 3400 3300]
    clustered_hotspots: Class distribution = [3300 3400 3300]
    random_pattern: Class distribution = [8589 1140  271]
    


```python
# Implementation
if __name__ == "__main__":
    # Train Group 1 (8 features for SHAP + LIME)
    trainer_group1 = train_models_workflow(
        datasets_group1, 
        group_name="Group 1 (SHAP + LIME)",
        save_prefix="group1"
    )
    
    # Train Group 2 (10 features for SHAP + GeoSHAP)
    trainer_group2 = train_models_workflow(
        datasets_group2,
        group_name="Group 2 (SHAP + GeoSHAP)",
        save_prefix="group2"
    )
```

    
    ======================================================================
    TRAINING Group 1 (SHAP + LIME) MODELS
    ======================================================================
    
    ============================================================
    TRAINING MODELS FOR ALL DATASETS
    ============================================================
    
    Training model for smooth_gradient...
    Training set shape: (8000, 8)
    Test set shape: (2000, 8)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">576</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.9695
    Test Loss: 0.0675
    Class distribution in test set: [660 680 660]
    Class distribution in predictions: [647 703 650]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.97      0.98       660
               1       0.94      0.97      0.96       680
               2       0.98      0.97      0.97       660
    
        accuracy                           0.97      2000
       macro avg       0.97      0.97      0.97      2000
    weighted avg       0.97      0.97      0.97      2000
    
    
    --------------------------------------------------
    
    Training model for sharp_boundary...
    Training set shape: (8000, 8)
    Test set shape: (2000, 8)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">576</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.6600
    Test Loss: 0.6452
    Class distribution in test set: [660 680 660]
    Class distribution in predictions: [1016    0  984]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.65      1.00      0.79       660
               1       0.00      0.00      0.00       680
               2       0.67      1.00      0.80       660
    
        accuracy                           0.66      2000
       macro avg       0.44      0.67      0.53      2000
    weighted avg       0.44      0.66      0.52      2000
    
    
    --------------------------------------------------
    
    Training model for clustered_hotspots...
    Training set shape: (8000, 8)
    Test set shape: (2000, 8)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">576</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.7760
    Test Loss: 0.7133
    Class distribution in test set: [660 680 660]
    Class distribution in predictions: [ 570 1021  409]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      0.86      0.93       660
               1       0.61      0.92      0.74       680
               2       0.87      0.54      0.66       660
    
        accuracy                           0.78      2000
       macro avg       0.83      0.77      0.78      2000
    weighted avg       0.83      0.78      0.78      2000
    
    
    --------------------------------------------------
    
    Training model for random_pattern...
    Training set shape: (8000, 8)
    Test set shape: (2000, 8)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">576</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">771</span> (3.01 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.8930
    Test Loss: 0.2951
    Class distribution in test set: [1718  228   54]
    Class distribution in predictions: [1778  209   13]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.94      0.97      0.95      1718
               1       0.54      0.50      0.52       228
               2       0.31      0.07      0.12        54
    
        accuracy                           0.89      2000
       macro avg       0.60      0.51      0.53      2000
    weighted avg       0.88      0.89      0.88      2000
    
    
    --------------------------------------------------
    
    ============================================================
    TRAINING SUMMARY
    ============================================================
    smooth_gradient: Accuracy = 0.9695
    sharp_boundary: Accuracy = 0.6600
    clustered_hotspots: Accuracy = 0.7760
    random_pattern: Accuracy = 0.8930
    


    
![png](output_20_25.png)
    


    
    ======================================================================
    TESTING LOGITS FUNCTIONALITY
    ======================================================================
    
    Testing Logits Functionality for smooth_gradient
    ==================================================
    Converting smooth_gradient model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
    Logits vs Probabilities Comparison for smooth_gradient:
    Sample shape: (5, 8)
    Logits shape: (5, 3)
    Probabilities shape: (5, 3)
    Max difference between P(softmax(logits)) and P(original): 1.69e-21
    
    Sample logits[0]: [-63.81262    -4.6411963  29.018328 ]
    Sample probabilities[0]: [0.0000000e+00 2.4091008e-15 1.0000000e+00]
    Sample softmax(logits)[0]: [4.8309765e-41 2.4091004e-15 1.0000000e+00]
    
    Additivity test with logits:
    Logits sum per sample: [-39.435486 -38.16078  -39.345634 -38.50505  -39.44541 ]
    Probabilities sum per sample: [1. 1. 1. 1. 1.]
    
    Testing gradient functionality:
    Gradients shape: (5, 8)
    Importances shape: (5, 8)
    Sample importance[0]: [13.776092   11.322918    5.766108    0.08427382  0.05904885  0.34369078
     -0.06008624 -0.02006715]
    Model saved: ./models/group1\smooth_gradient_model.keras
    Model saved: ./models/group1\sharp_boundary_model.keras
    Model saved: ./models/group1\clustered_hotspots_model.keras
    Model saved: ./models/group1\random_pattern_model.keras
    
    Group 1 (SHAP + LIME) training completed. Models trained: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
    
    ======================================================================
    TRAINING Group 2 (SHAP + GeoSHAP) MODELS
    ======================================================================
    
    ============================================================
    TRAINING MODELS FOR ALL DATASETS
    ============================================================
    
    Training model for smooth_gradient...
    Training set shape: (8000, 10)
    Test set shape: (2000, 10)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">704</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.9700
    Test Loss: 0.0673
    Class distribution in test set: [660 680 660]
    Class distribution in predictions: [655 696 649]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.99      0.98      0.98       660
               1       0.95      0.97      0.96       680
               2       0.98      0.96      0.97       660
    
        accuracy                           0.97      2000
       macro avg       0.97      0.97      0.97      2000
    weighted avg       0.97      0.97      0.97      2000
    
    
    --------------------------------------------------
    
    Training model for sharp_boundary...
    Training set shape: (8000, 10)
    Test set shape: (2000, 10)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">704</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.6600
    Test Loss: 0.6457
    Class distribution in test set: [660 680 660]
    Class distribution in predictions: [1016    0  984]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.65      1.00      0.79       660
               1       0.00      0.00      0.00       680
               2       0.67      1.00      0.80       660
    
        accuracy                           0.66      2000
       macro avg       0.44      0.67      0.53      2000
    weighted avg       0.44      0.66      0.52      2000
    
    
    --------------------------------------------------
    
    Training model for clustered_hotspots...
    Training set shape: (8000, 10)
    Test set shape: (2000, 10)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">704</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.8195
    Test Loss: 0.3825
    Class distribution in test set: [660 680 660]
    Class distribution in predictions: [616 581 803]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      0.93      0.97       660
               1       0.77      0.66      0.71       680
               2       0.71      0.87      0.78       660
    
        accuracy                           0.82      2000
       macro avg       0.83      0.82      0.82      2000
    weighted avg       0.83      0.82      0.82      2000
    
    
    --------------------------------------------------
    
    Training model for random_pattern...
    Training set shape: (8000, 10)
    Test set shape: (2000, 10)
    
    Model architecture:
    


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "mlp_classifier"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                         </span>┃<span style="font-weight: bold"> Output Shape                </span>┃<span style="font-weight: bold">         Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ hidden_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │             <span style="color: #00af00; text-decoration-color: #00af00">704</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)                    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)                  │               <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ output_layer (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)                   │             <span style="color: #00af00; text-decoration-color: #00af00">195</span> │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">899</span> (3.51 KB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">0</span> (0.00 B)
</pre>



    
    Test Accuracy: 0.9110
    Test Loss: 0.2482
    Class distribution in test set: [1718  228   54]
    Class distribution in predictions: [1737  262    1]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.96      0.97      0.96      1718
               1       0.60      0.68      0.64       228
               2       0.00      0.00      0.00        54
    
        accuracy                           0.91      2000
       macro avg       0.52      0.55      0.53      2000
    weighted avg       0.89      0.91      0.90      2000
    
    
    --------------------------------------------------
    
    ============================================================
    TRAINING SUMMARY
    ============================================================
    smooth_gradient: Accuracy = 0.9700
    sharp_boundary: Accuracy = 0.6600
    clustered_hotspots: Accuracy = 0.8195
    random_pattern: Accuracy = 0.9110
    


    
![png](output_20_51.png)
    


    
    ======================================================================
    TESTING LOGITS FUNCTIONALITY
    ======================================================================
    
    Testing Logits Functionality for smooth_gradient
    ==================================================
    Converting smooth_gradient model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
    Logits vs Probabilities Comparison for smooth_gradient:
    Sample shape: (5, 10)
    Logits shape: (5, 3)
    Probabilities shape: (5, 3)
    Max difference between P(softmax(logits)) and P(original): 5.29e-23
    
    Sample logits[0]: [-36.685112  -6.222141  30.861126]
    Sample probabilities[0]: [4.6242718e-30 7.8513019e-17 1.0000000e+00]
    Sample softmax(logits)[0]: [4.6242703e-30 7.8513019e-17 1.0000000e+00]
    
    Additivity test with logits:
    Logits sum per sample: [-12.046127  -11.5021515 -11.898731  -11.662952  -12.080151 ]
    Probabilities sum per sample: [1. 1. 1. 1. 1.]
    
    Testing gradient functionality:
    Gradients shape: (5, 10)
    Importances shape: (5, 10)
    Sample importance[0]: [ 1.4078286e+01  1.1516663e+01  6.4531565e+00  7.2188243e-02
      6.0797825e-02  2.2785348e-01 -2.3025135e-02 -5.1541310e-03
     -0.0000000e+00 -0.0000000e+00]
    Model saved: ./models/group2\smooth_gradient_model.keras
    Model saved: ./models/group2\sharp_boundary_model.keras
    Model saved: ./models/group2\clustered_hotspots_model.keras
    Model saved: ./models/group2\random_pattern_model.keras
    
    Group 2 (SHAP + GeoSHAP) training completed. Models trained: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
    


```python
class SampleSelectionAndReference:
    """Handles consistent sample selection across ROIs for XAI analysis"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def select_one_sample_per_roi(self, dataset_dict):
        """
        Select exactly one sample per ROI consistently
        
        Parameters:
        - dataset_dict: Dataset dictionary with X, y, roi_labels
        
        Returns:
        - selected_samples: Dict with X_selected, y_selected, roi_indices, sample_indices
        """
        X = dataset_dict['X']
        y = dataset_dict['y'] 
        roi_labels = dataset_dict['roi_labels']
        n_rois = dataset_dict['grid_size'] ** 2
        
        # Store results
        X_selected = []
        y_selected = []
        roi_indices = []
        sample_indices = []  # Original indices in the dataset
        
        # Set seed for consistent selection
        rng = np.random.RandomState(self.random_state)
        
        for roi_idx in range(n_rois):
            roi_mask = roi_labels == roi_idx
            roi_sample_indices = np.where(roi_mask)[0]
            
            if len(roi_sample_indices) > 0:
                # Select one random sample from this ROI
                selected_idx = rng.choice(roi_sample_indices)
                
                X_selected.append(X[selected_idx])
                y_selected.append(y[selected_idx])
                roi_indices.append(roi_idx)
                sample_indices.append(selected_idx)
        
        return {
            'X_selected': np.array(X_selected),
            'y_selected': np.array(y_selected), 
            'roi_indices': np.array(roi_indices),
            'sample_indices': np.array(sample_indices),
            'n_samples': len(X_selected),
            'dataset_name': dataset_dict['dataset_name']
        }
    
    def get_ground_truth_importance(self, dataset_dict, selected_samples):
        """
        Calculate ground truth feature importance using roi_feature_weights
        
        Parameters:
        - dataset_dict: Original dataset
        - selected_samples: Result from select_one_sample_per_roi
        
        Returns:
        - ground_truth_importance: Array of shape (n_rois, n_features)
        """
        roi_feature_weights = dataset_dict.get('roi_feature_weights', None)
        
        if roi_feature_weights is None:
            raise ValueError("Dataset must contain roi_feature_weights for ground truth")
        
        # Extract weights for selected ROIs
        roi_indices = selected_samples['roi_indices']
        ground_truth = np.array([roi_feature_weights[roi_idx] for roi_idx in roi_indices])
        
        return ground_truth
    
    def get_gradient_reference_importance(self, trainer, dataset_name, selected_samples):
        """
        Calculate gradient × input reference importance for selected samples
        
        Parameters:
        - trainer: Trained SimpleMLPModels instance
        - dataset_name: Name of dataset/model
        - selected_samples: Result from select_one_sample_per_roi
        
        Returns:
        - gradient_importance: Array of shape (n_samples, n_features)
        """
        X_selected = selected_samples['X_selected']
        
        # Get gradient × input importance
        gradients, importance = trainer.get_feature_importance_gradients(
            dataset_name, X_selected
        )
        
        return importance


def setup_consistent_samples_for_analysis(datasets, trainer):
    """
    Set up consistent sample selection for all datasets
    
    Parameters:
    - datasets: Dict of {name: dataset_dict}
    - trainer: Trained SimpleMLPModels instance
    
    Returns:
    - analysis_data: Dict with selected samples and reference importance for each dataset
    """
    selector = SampleSelectionAndReference(random_state=42)
    analysis_data = {}
    
    print("Setting up consistent samples for XAI analysis...")
    
    for name, dataset in datasets.items():
        print(f"\nProcessing {name}:")
        
        # Select one sample per ROI
        selected_samples = selector.select_one_sample_per_roi(dataset)
        print(f"  Selected {selected_samples['n_samples']} samples (one per ROI)")
        
        # Get ground truth importance (from dataset generation)
        try:
            ground_truth = selector.get_ground_truth_importance(dataset, selected_samples)
            print(f"  Ground truth importance shape: {ground_truth.shape}")
        except ValueError as e:
            print(f"  Warning: {e}")
            ground_truth = None
        
        # Get gradient reference importance (from trained MLP)
        gradient_ref = selector.get_gradient_reference_importance(
            trainer, name, selected_samples
        )
        print(f"  Gradient reference importance shape: {gradient_ref.shape}")
        
        analysis_data[name] = {
            'selected_samples': selected_samples,
            'ground_truth_importance': ground_truth,
            'gradient_reference_importance': gradient_ref,
            'dataset': dataset  # Keep reference to original
        }
    
    return analysis_data
```


```python
# ============================================================================
# SAMPLE SELECTION AND REFERENCE SETUP
# ============================================================================

print("\n" + "="*70)
print("SETTING UP SAMPLES FOR XAI ANALYSIS")
print("="*70)

# Group 1: 8-variable datasets (SHAP + LIME)
print("\n### GROUP 1 ###")
analysis_data_group1 = setup_consistent_samples_for_analysis(datasets_group1, trainer_group1)

# Group 2: 10-variable datasets (SHAP + GeoSHAP)
print("\n### GROUP 2 ###")
analysis_data_group2 = setup_consistent_samples_for_analysis(datasets_group2, trainer_group2)

print("\n" + "="*70)
print("SAMPLE SELECTION COMPLETE")
print("="*70)

# Verify the setup
print("\n### VERIFICATION ###")
print("\nGroup 1 Analysis Data:")
for name in analysis_data_group1.keys():
    data = analysis_data_group1[name]
    print(f"  {name}:")
    print(f"    Selected samples: {data['selected_samples']['n_samples']}")
    print(f"    Ground truth shape: {data['ground_truth_importance'].shape if data['ground_truth_importance'] is not None else 'None'}")
    print(f"    Gradient reference shape: {data['gradient_reference_importance'].shape}")

print("\nGroup 2 Analysis Data:")
for name in analysis_data_group2.keys():
    data = analysis_data_group2[name]
    print(f"  {name}:")
    print(f"    Selected samples: {data['selected_samples']['n_samples']}")
    print(f"    Ground truth shape: {data['ground_truth_importance'].shape if data['ground_truth_importance'] is not None else 'None'}")
    print(f"    Gradient reference shape: {data['gradient_reference_importance'].shape}")
```

    
    ======================================================================
    SETTING UP SAMPLES FOR XAI ANALYSIS
    ======================================================================
    
    ### GROUP 1 ###
    Setting up consistent samples for XAI analysis...
    
    Processing smooth_gradient:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
      Gradient reference importance shape: (100, 8)
    
    Processing sharp_boundary:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
    Converting sharp_boundary model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
      Gradient reference importance shape: (100, 8)
    
    Processing clustered_hotspots:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
    Converting clustered_hotspots model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
      Gradient reference importance shape: (100, 8)
    
    Processing random_pattern:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
    Converting random_pattern model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
      Gradient reference importance shape: (100, 8)
    
    ### GROUP 2 ###
    Setting up consistent samples for XAI analysis...
    
    Processing smooth_gradient:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
      Gradient reference importance shape: (100, 10)
    
    Processing sharp_boundary:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
    Converting sharp_boundary model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
      Gradient reference importance shape: (100, 10)
    
    Processing clustered_hotspots:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
    Converting clustered_hotspots model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
      Gradient reference importance shape: (100, 10)
    
    Processing random_pattern:
      Selected 100 samples (one per ROI)
      Ground truth importance shape: (100, 8)
    Converting random_pattern model to logits output...
    ✓ Logits model created: softmax → linear
      Original activation: softmax
      New activation: linear
      Gradient reference importance shape: (100, 10)
    
    ======================================================================
    SAMPLE SELECTION COMPLETE
    ======================================================================
    
    ### VERIFICATION ###
    
    Group 1 Analysis Data:
      smooth_gradient:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 8)
      sharp_boundary:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 8)
      clustered_hotspots:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 8)
      random_pattern:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 8)
    
    Group 2 Analysis Data:
      smooth_gradient:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 10)
      sharp_boundary:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 10)
      clustered_hotspots:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 10)
      random_pattern:
        Selected samples: 100
        Ground truth shape: (100, 8)
        Gradient reference shape: (100, 10)
    

## Create SHAP, GeoSHAP, and LIME XAI


```python
import numpy as np
import itertools
from scipy.special import comb
from tqdm import tqdm
import matplotlib.pyplot as plt
```


```python
"""
Standard SHAP Explainer
Overall, not by ROI
Treats all features independently (no joint spatial treatment)
Suitable for both Group 1 (8 variables) and Group 2 (10 variables)
"""

import numpy as np
import itertools
from scipy.special import comb
from tqdm import tqdm
import matplotlib.pyplot as plt


class StandardSHAPExplainer:
    """
    Standard SHAP implementation using exact Shapley values
    All features are treated independently
    
    This is used as a baseline for both:
    - Group 1 (8 variables): Standard SHAP for comparison with LIME
    - Group 2 (10 variables): Standard SHAP for comparison with GeoSHAP
    """
    
    def __init__(self, trainer, dataset_name, X_data, background_type='mean', 
                 mode='decision_aligned', roi_labels=None, coords=None, grid_size=None):
        """
        Initialize Standard SHAP Explainer
        
        Parameters:
        -----------
        trainer : SimpleMLPModels instance
            Trained model wrapper with logits methods
        dataset_name : str
            Name of dataset/model (e.g., 'smooth_gradient')
        X_data : np.ndarray
            Feature data (can be 8 or 10 features)
        background_type : str
            'mean' for global mean background or 'queen' for spatial neighbors
        mode : str
            'decision_aligned' or 'class_specific'
        roi_labels : np.ndarray, optional
            ROI labels for queen background (needed if background_type='queen')
        coords : np.ndarray, optional
            Coordinate array for ROIs (needed if background_type='queen')
        grid_size : int, optional
            Size of spatial grid (needed if background_type='queen')
        """
        self.trainer = trainer
        self.dataset_name = dataset_name
        self.model = trainer.get_model(dataset_name)
        
        if self.model is None:
            raise ValueError(f"No trained model found for dataset: {dataset_name}")
        
        self.X_data = X_data
        self.n_features = X_data.shape[1]
        self.n_classes = self.model.output_shape[1]
        self.background_type = background_type
        self.mode = mode
        
        # Spatial information (optional, only needed for queen background)
        self.roi_labels = roi_labels
        self.coords = coords
        self.grid_size = grid_size
        
        # Compute backgrounds
        self.backgrounds = self._compute_backgrounds()
        
        print(f"StandardSHAPExplainer initialized for {dataset_name}")
        print(f"  Features: {self.n_features}, Classes: {self.n_classes}")
        print(f"  Background: {background_type}, Mode: {mode}")
    
    def _compute_backgrounds(self):
        """Compute background values based on background_type"""
        backgrounds = {}
        
        if self.background_type == 'mean':
            # Global mean background - same for all samples
            global_mean = np.mean(self.X_data, axis=0)
            
            if self.roi_labels is not None:
                for roi_id in np.unique(self.roi_labels):
                    backgrounds[roi_id] = global_mean
            else:
                # If no ROI labels, use sample indices as keys
                for i in range(len(self.X_data)):
                    backgrounds[i] = global_mean
                    
        elif self.background_type == 'queen':
            if self.roi_labels is None or self.coords is None or self.grid_size is None:
                raise ValueError("roi_labels, coords, and grid_size required for queen background")
            
            # Queen configuration - 8 surrounding neighbors
            for roi_id in np.unique(self.roi_labels):
                queen_samples = self._get_queen_neighbors(roi_id)
                if len(queen_samples) > 0:
                    backgrounds[roi_id] = np.mean(queen_samples, axis=0)
                else:
                    # Fallback to global mean
                    backgrounds[roi_id] = np.mean(self.X_data, axis=0)
        
        return backgrounds
    
    def _get_queen_neighbors(self, roi_id):
        """Get samples from 8 surrounding neighbors (queen configuration)"""
        row, col = self.coords[roi_id]
        neighbor_samples = []
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        directions = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
        
        for dr, dc in directions:
            neighbor_row, neighbor_col = row + dr, col + dc
            
            if 0 <= neighbor_row < self.grid_size and 0 <= neighbor_col < self.grid_size:
                neighbor_roi_id = neighbor_row * self.grid_size + neighbor_col
                neighbor_mask = self.roi_labels == neighbor_roi_id
                neighbor_data = self.X_data[neighbor_mask]
                
                if len(neighbor_data) > 0:
                    neighbor_samples.append(neighbor_data)
        
        if neighbor_samples:
            return np.vstack(neighbor_samples)
        else:
            return np.array([])
    
    def _get_background_for_sample(self, sample_idx):
        """Get appropriate background for a sample"""
        if self.roi_labels is not None:
            roi_id = self.roi_labels[sample_idx]
            return self.backgrounds[roi_id]
        else:
            return self.backgrounds[sample_idx]
    
    def _get_model_output(self, X, output_type='logits'):
        """Get model output as logits or probabilities"""
        if output_type == 'logits':
            return self.trainer.get_logits_prediction(self.dataset_name, X, verbose=0)
        else:
            return self.trainer.get_probabilities_prediction(self.dataset_name, X, verbose=0)
    
    def _get_all_coalitions(self):
        """Pre-generate all possible coalitions for efficiency"""
        if not hasattr(self, '_coalitions_cache'):
            print("Pre-computing coalition structures...")
            self._coalitions_cache = {}
            
            for feature_idx in range(self.n_features):
                other_features = [i for i in range(self.n_features) if i != feature_idx]
                coalitions = []
                
                for size in range(len(other_features) + 1):
                    coalitions.extend(list(itertools.combinations(other_features, size)))
                
                self._coalitions_cache[feature_idx] = coalitions
            
            total_coalitions = sum(len(v) for v in self._coalitions_cache.values())
            print(f"Cached {total_coalitions} coalition patterns")
        
        return self._coalitions_cache
    
    def _get_coalition_values_batch(self, sample, coalitions, background, target_class):
        """Batch process multiple coalitions for efficiency"""
        batch_samples = []
        
        for coalition in coalitions:
            coalition_sample = background.copy()
            coalition_sample[list(coalition)] = sample[list(coalition)]
            batch_samples.append(coalition_sample)
        
        if len(batch_samples) == 0:
            return np.array([])
        
        batch_array = np.array(batch_samples)
        batch_outputs = self._get_model_output(batch_array, output_type='logits')
        
        return batch_outputs[:, target_class]
    
    def _exact_shap_values(self, sample, background, target_class):
        """Compute exact SHAP values using all possible coalitions"""
        shap_values = np.zeros(self.n_features)
        coalitions_cache = self._get_all_coalitions()
        
        # For each feature
        for feature_idx in range(self.n_features):
            coalitions = coalitions_cache[feature_idx]
            
            # Batch process coalitions without the feature
            coalitions_without = coalitions
            values_without = self._get_coalition_values_batch(
                sample, coalitions_without, background, target_class
            )
            
            # Batch process coalitions with the feature added
            coalitions_with = [list(coalition) + [feature_idx] for coalition in coalitions]
            values_with = self._get_coalition_values_batch(
                sample, coalitions_with, background, target_class
            )
            
            # Calculate marginal contributions
            marginal_contributions = values_with - values_without
            
            # Apply Shapley weights
            weighted_contributions = []
            for i, coalition in enumerate(coalitions):
                coalition_size = len(coalition)
                weight = 1.0 / comb(self.n_features - 1, coalition_size, exact=True)
                weighted_contributions.append(weight * marginal_contributions[i])
            
            shap_values[feature_idx] = np.sum(weighted_contributions) / self.n_features
        
        return shap_values
    
    def explain_sample(self, sample_idx, target_class=None):
        """
        Compute SHAP values for a single sample
        
        Parameters:
        -----------
        sample_idx : int
            Index of sample to explain
        target_class : int, optional
            Class to explain (auto-detected if None in decision_aligned mode)
        
        Returns:
        --------
        dict with keys:
            - shap_values: SHAP values for each feature
            - baseline: Baseline value
            - full_value: Full coalition value
            - predicted_full: Baseline + sum of SHAP values
            - additivity_error: |predicted_full - full_value|
            - target_class: Class being explained
        """
        sample = self.X_data[sample_idx]
        background = self._get_background_for_sample(sample_idx)
        
        # Auto-detect target class for decision-aligned mode
        if self.mode == 'decision_aligned' and target_class is None:
            sample_output = self._get_model_output(sample.reshape(1, -1), output_type='logits')
            target_class = np.argmax(sample_output[0])
        
        # Get baseline value (empty coalition)
        baseline_sample = background.reshape(1, -1)
        baseline_output = self._get_model_output(baseline_sample, output_type='logits')
        baseline = baseline_output[0, target_class]
        
        # Get full coalition value
        full_sample = sample.reshape(1, -1)
        full_output = self._get_model_output(full_sample, output_type='logits')
        full_value = full_output[0, target_class]
        
        # Compute SHAP values
        shap_values = self._exact_shap_values(sample, background, target_class)
        
        # Verify additivity
        predicted_full = baseline + np.sum(shap_values)
        additivity_error = abs(predicted_full - full_value)
        
        return {
            'shap_values': shap_values,
            'baseline': baseline,
            'full_value': full_value,
            'predicted_full': predicted_full,
            'additivity_error': additivity_error,
            'target_class': target_class
        }
    
    def explain_samples(self, sample_indices, target_class=None):
        """
        Compute SHAP values for multiple samples
        
        Parameters:
        -----------
        sample_indices : list or np.ndarray
            Indices of samples to explain
        target_class : int, optional
            Class to explain (for class_specific mode)
        
        Returns:
        --------
        dict with keys:
            - shap_values: Array of SHAP values (n_samples, n_features)
            - individual_results: List of individual result dicts
            - mean_additivity_error: Mean additivity error
            - max_additivity_error: Max additivity error
        """
        print(f"Computing SHAP values for {len(sample_indices)} samples...")
        
        all_shap_values = []
        all_results = []
        
        for sample_idx in tqdm(sample_indices, desc="Computing SHAP"):
            if self.mode == 'class_specific' and target_class is None:
                # For class-specific mode, use predicted class
                sample_output = self._get_model_output(
                    self.X_data[sample_idx:sample_idx+1], output_type='logits'
                )
                current_target = np.argmax(sample_output[0])
                result = self.explain_sample(sample_idx, target_class=current_target)
            else:
                result = self.explain_sample(sample_idx, target_class=target_class)
            
            all_shap_values.append(result['shap_values'])
            all_results.append(result)
        
        shap_array = np.array(all_shap_values)
        
        return {
            'shap_values': shap_array,
            'individual_results': all_results,
            'mean_additivity_error': np.mean([r['additivity_error'] for r in all_results]),
            'max_additivity_error': np.max([r['additivity_error'] for r in all_results]),
            'configuration': (self.background_type, self.mode)
        }
    
    def visualize_feature_importance(self, shap_results, feature_names=None):
        """
        Visualize SHAP feature importance
        
        Parameters:
        -----------
        shap_results : dict
            Results from explain_samples()
        feature_names : list, optional
            Names of features
        """
        shap_values = shap_results['shap_values']
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(self.n_features)]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Sort features by importance
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(self.n_features), mean_abs_shap[sorted_idx])
        plt.yticks(range(self.n_features), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Feature Importance - {self.dataset_name}\n{self.background_type} background, {self.mode}')
        plt.tight_layout()
        plt.show()
```


```python
# =============================================================================
# 1. SETUP
# =============================================================================

# Choose dataset
dataset_name = 'smooth_gradient'  # Change as needed

# Get dataset and selected samples from YOUR existing variables
dataset = datasets_group1[dataset_name]  # Use datasets_group2 for Group 2
analysis_data = analysis_data_group1[dataset_name]  # Use analysis_data_group2 for Group 2
selected = analysis_data['selected_samples']

# Initialize explainer
shap_explainer = StandardSHAPExplainer(
    trainer=trainer_group1,  # Use trainer_group2 for Group 2
    dataset_name=dataset_name,
    X_data=dataset['X'],
    background_type='mean',
    mode='decision_aligned'
)

# =============================================================================
# 2. COMPUTE SHAP VALUES
# =============================================================================

print(f"Computing SHAP for {len(selected['sample_indices'])} samples...")

shap_results = shap_explainer.explain_samples(
    sample_indices=selected['sample_indices']
)

shap_values = shap_results['shap_values']
print(f"Done. Shape: {shap_values.shape}")
print(f"Additivity error: {shap_results['mean_additivity_error']:.2e}")

# =============================================================================
# 3. FEATURE IMPORTANCE
# =============================================================================

mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

plt.figure(figsize=(10, 5))
plt.barh(range(len(mean_abs_shap)), mean_abs_shap)
plt.yticks(range(len(mean_abs_shap)), [f'Feature {i}' for i in range(len(mean_abs_shap))])
plt.xlabel('Mean |SHAP value|')
plt.title(f'{dataset_name} - Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =============================================================================
# 4. SAVE RESULTS
# =============================================================================

# Save for later use
standard_shap_results = {
    'shap_values': shap_values,
    'mean_abs_shap': mean_abs_shap,
    'results': shap_results
}

print("Results saved in 'standard_shap_results'")
```

    StandardSHAPExplainer initialized for smooth_gradient
      Features: 8, Classes: 3
      Background: mean, Mode: decision_aligned
    Computing SHAP for 100 samples...
    Computing SHAP values for 100 samples...
    

    Computing SHAP:   0%|                                                                          | 0/100 [00:00<?, ?it/s]

    Pre-computing coalition structures...
    Cached 1024 coalition patterns
    

    Computing SHAP: 100%|████████████████████████████████████████████████████████████████| 100/100 [04:41<00:00,  2.82s/it]
    

    Done. Shape: (100, 8)
    Additivity error: 1.39e-06
    


    
![png](output_26_5.png)
    


    Results saved in 'standard_shap_results'
    


```python
class CustomSHAPExplainer:
    """
    Custom SHAP implementation from scratch for spatial XAI framework
    Supports both mean and queen configuration backgrounds with decision-aligned and class-specific modes
    Uses exact SHAP computation only with batching optimizations
    Creates SHAP values for each ROI
    """
    
    def __init__(self, trainer, dataset_name, X_data, roi_labels, coords, grid_size, background_type='mean', mode='decision_aligned'):
        """
        Initialize Custom SHAP Explainer
        
        Parameters:
        - trainer: SimpleMLPModels instance with logits methods
        - dataset_name: Name of dataset/model to use (e.g., 'smooth_gradient')
        - X_data: Feature data (normalized)
        - roi_labels: ROI labels for each sample
        - coords: Coordinate array for ROIs
        - grid_size: Size of spatial grid
        - background_type: 'mean' or 'queen' 
        - mode: 'decision_aligned' or 'class_specific'
        """
        self.trainer = trainer
        self.dataset_name = dataset_name
        self.model = trainer.get_model(dataset_name)
        if self.model is None:
            raise ValueError(f"No trained model found for dataset: {dataset_name}")
            
        self.X_data = X_data
        self.roi_labels = roi_labels
        self.coords = coords
        self.grid_size = grid_size
        self.background_type = background_type
        self.mode = mode
        self.n_features = X_data.shape[1]
        self.n_classes = self.model.output_shape[1]
        
        # Store background references for each ROI
        self.backgrounds = self._compute_backgrounds()
        
        print(f"  CustomSHAPExplainer initialized for {dataset_name}")
        print(f"  Background type: {background_type}")
        print(f"  Mode: {mode}")
        print(f"  Features: {self.n_features}, Classes: {self.n_classes}")
        
    def _compute_backgrounds(self):
        """Compute background values for each ROI based on background_type"""
        backgrounds = {}
        
        if self.background_type == 'mean':
            # Global mean background - same for all ROIs
            global_mean = np.mean(self.X_data, axis=0)
            for roi_id in np.unique(self.roi_labels):
                backgrounds[roi_id] = global_mean
                
        elif self.background_type == 'queen':
            # Queen configuration - 8 surrounding neighbors
            for roi_id in np.unique(self.roi_labels):
                queen_samples = self._get_queen_neighbors(roi_id)
                if len(queen_samples) > 0:
                    backgrounds[roi_id] = np.mean(queen_samples, axis=0)
                else:
                    # Fallback to global mean if no neighbors
                    backgrounds[roi_id] = np.mean(self.X_data, axis=0)
        
        return backgrounds
    
    def _get_queen_neighbors(self, roi_id):
        """Get samples from 8 surrounding neighbors (queen configuration)"""
        row, col = self.coords[roi_id]
        neighbor_samples = []
        
        # 8 directions: N, NE, E, SE, S, SW, W, NW
        directions = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
        
        for dr, dc in directions:
            neighbor_row, neighbor_col = row + dr, col + dc
            
            # Check bounds
            if 0 <= neighbor_row < self.grid_size and 0 <= neighbor_col < self.grid_size:
                neighbor_roi_id = neighbor_row * self.grid_size + neighbor_col
                neighbor_mask = self.roi_labels == neighbor_roi_id
                neighbor_data = self.X_data[neighbor_mask]
                
                if len(neighbor_data) > 0:
                    neighbor_samples.append(neighbor_data)
        
        if neighbor_samples:
            return np.vstack(neighbor_samples)
        else:
            return np.array([])
    
    def _get_model_output(self, X, output_type='logits'):
        """Get model output as logits or probabilities using trainer methods"""
        if output_type == 'logits':
            return self.trainer.get_logits_prediction(self.dataset_name, X, verbose=0)
        else:
            return self.trainer.get_probabilities_prediction(self.dataset_name, X, verbose=0)
    
    def _get_all_coalitions(self):
        """Pre-generate all possible coalitions once for efficiency"""
        if not hasattr(self, '_coalitions_cache'):
            print("Pre-computing coalition structures...")
            self._coalitions_cache = {}
            for feature_idx in range(self.n_features):
                other_features = [i for i in range(self.n_features) if i != feature_idx]
                coalitions = []
                for size in range(len(other_features) + 1):
                    coalitions.extend(list(itertools.combinations(other_features, size)))
                self._coalitions_cache[feature_idx] = coalitions
            print(f"Cached {sum(len(v) for v in self._coalitions_cache.values())} coalition patterns")
        return self._coalitions_cache
    
    def _get_coalition_values_batch(self, sample, coalitions, background, target_class):
        """Batch process multiple coalitions for efficiency"""
        batch_samples = []
        for coalition in coalitions:
            coalition_sample = background.copy()
            coalition_sample[list(coalition)] = sample[list(coalition)]
            batch_samples.append(coalition_sample)
        
        if len(batch_samples) == 0:
            return np.array([])
        
        batch_array = np.array(batch_samples)
        batch_outputs = self._get_model_output(batch_array, output_type='logits')
        
        return batch_outputs[:, target_class]
    
    def _get_coalition_value(self, sample, coalition, background, target_class=None):
        """
        Calculate the coalition value f(S ∪ {i}) or f(S) for a given coalition
        
        Parameters:
        - sample: Original sample
        - coalition: List of feature indices in the coalition
        - background: Background values for this ROI
        - target_class: Class to explain (None for decision-aligned mode)
        """
        # Create coalition sample: use original features for coalition, background for others
        coalition_sample = background.copy()
        coalition_sample[coalition] = sample[coalition]
        
        # Get model prediction for this coalition
        coalition_sample = coalition_sample.reshape(1, -1)
        output = self._get_model_output(coalition_sample, output_type='logits')
        
        if self.mode == 'decision_aligned':
            # Use the predicted class
            if target_class is None:
                target_class = np.argmax(output[0])
            return output[0, target_class]
        else:  # class_specific
            # Return output for specified class
            if target_class is None:
                raise ValueError("target_class must be specified for class_specific mode")
            return output[0, target_class]
    
    def explain_sample(self, sample_idx, target_class=None):
        """
        Compute SHAP values for a single sample using exact Shapley values
        
        Parameters:
        - sample_idx: Index of sample to explain
        - target_class: Class to explain (auto-detected if None in decision_aligned mode)
        
        Returns:
        - shap_values: SHAP values for each feature
        - baseline: Baseline (background) value
        """
        sample = self.X_data[sample_idx]
        roi_id = self.roi_labels[sample_idx]
        background = self.backgrounds[roi_id]
        
        # Auto-detect target class for decision-aligned mode
        if self.mode == 'decision_aligned' and target_class is None:
            sample_output = self._get_model_output(sample.reshape(1, -1), output_type='logits')
            target_class = np.argmax(sample_output[0])
        
        # Get baseline value (empty coalition)
        baseline = self._get_coalition_value(sample, [], background, target_class)
        
        # Get full coalition value
        full_coalition = list(range(self.n_features))
        full_value = self._get_coalition_value(sample, full_coalition, background, target_class)
        
        # Use exact SHAP computation
        shap_values = self._exact_shap_values(sample, background, target_class, baseline)
        
        # Verify additivity
        predicted_full = baseline + np.sum(shap_values)
        actual_full = full_value
        additivity_error = abs(predicted_full - actual_full)
        
        return {
            'shap_values': shap_values,
            'baseline': baseline,
            'full_value': full_value,
            'predicted_full': predicted_full,
            'additivity_error': additivity_error,
            'target_class': target_class,
            'roi_id': roi_id
        }
    
    def _exact_shap_values(self, sample, background, target_class, baseline):
        """Compute exact SHAP values using all possible coalitions with batching for efficiency"""
        shap_values = np.zeros(self.n_features)
        coalitions_cache = self._get_all_coalitions()
        
        # For each feature
        for feature_idx in range(self.n_features):
            coalitions = coalitions_cache[feature_idx]
            
            # Batch process coalitions without the feature
            coalitions_without = coalitions
            values_without = self._get_coalition_values_batch(sample, coalitions_without, background, target_class)
            
            # Batch process coalitions with the feature added
            coalitions_with = [list(coalition) + [feature_idx] for coalition in coalitions]
            values_with = self._get_coalition_values_batch(sample, coalitions_with, background, target_class)
            
            # Calculate marginal contributions
            marginal_contributions = values_with - values_without
            
            # Apply Shapley weights
            weighted_contributions = []
            for i, coalition in enumerate(coalitions):
                coalition_size = len(coalition)
                weight = 1.0 / comb(self.n_features - 1, coalition_size, exact=True)
                weighted_contributions.append(weight * marginal_contributions[i])
            
            shap_values[feature_idx] = np.sum(weighted_contributions) / self.n_features
        
        return shap_values
    
    def explain_roi(self, roi_id, n_samples=None, target_class=None):
        """
        Compute SHAP values for all samples in an ROI and return summary statistics
        
        Parameters:
        - roi_id: ROI identifier
        - n_samples: Number of samples to explain (None for all)
        - target_class: Class to explain (for class_specific mode)
        """
        # Get samples for this ROI
        roi_mask = self.roi_labels == roi_id
        roi_sample_indices = np.where(roi_mask)[0]
        
        if n_samples is not None:
            roi_sample_indices = np.random.choice(roi_sample_indices, 
                                                 min(n_samples, len(roi_sample_indices)), 
                                                 replace=False)
        
        roi_results = []
        
        print(f"Computing exact SHAP for {len(roi_sample_indices)} samples in ROI {roi_id}")
        for sample_idx in tqdm(roi_sample_indices, desc=f"ROI {roi_id}"):
            result = self.explain_sample(sample_idx, target_class)
            roi_results.append(result)
        
        # Aggregate results
        shap_values_array = np.array([r['shap_values'] for r in roi_results])
        
        summary = {
            'roi_id': roi_id,
            'n_samples': len(roi_results),
            'mean_shap_values': np.mean(shap_values_array, axis=0),
            'std_shap_values': np.std(shap_values_array, axis=0),
            'mean_baseline': np.mean([r['baseline'] for r in roi_results]),
            'mean_additivity_error': np.mean([r['additivity_error'] for r in roi_results]),
            'max_additivity_error': np.max([r['additivity_error'] for r in roi_results]),
            'target_classes': [r['target_class'] for r in roi_results],
            'all_shap_values': shap_values_array,
            'individual_results': roi_results
        }
        
        return summary
    
    def explain_all_rois(self, n_samples_per_roi=1, target_class=None):
        """
        Compute exact SHAP values for samples across all ROIs
        
        Parameters:
        - n_samples_per_roi: Number of samples to explain per ROI
        - target_class: Class to explain (for class_specific mode)
        
        Returns aggregated results suitable for spatial analysis
        """
        all_roi_results = []
        
        unique_rois = np.unique(self.roi_labels)
        print(f"Computing exact SHAP for {len(unique_rois)} ROIs with {n_samples_per_roi} samples each")
        
        for roi_id in unique_rois:
            roi_summary = self.explain_roi(roi_id, n_samples_per_roi, target_class)
            all_roi_results.append(roi_summary)
        
        return all_roi_results
    
    def visualize_shap_spatial_pattern(self, roi_results, feature_idx, title_prefix=""):
        """
        Visualize spatial pattern of SHAP values for a specific feature
        
        Parameters:
        - roi_results: Results from explain_all_rois
        - feature_idx: Feature to visualize
        - title_prefix: Prefix for plot title
        """
        # Create spatial grid of mean SHAP values
        shap_grid = np.zeros((self.grid_size, self.grid_size))
        
        for roi_result in roi_results:
            roi_id = roi_result['roi_id']
            row, col = self.coords[roi_id]
            shap_grid[row, col] = roi_result['mean_shap_values'][feature_idx]
        
        plt.figure(figsize=(8, 6))
        im = plt.imshow(shap_grid, cmap='RdBu_r', interpolation='nearest')
        plt.colorbar(im, label='Mean SHAP Value')
        plt.title(f'{title_prefix}Feature {feature_idx} SHAP Values - {self.background_type} background')
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        # Add text annotations
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                plt.text(j, i, f'{shap_grid[i, j]:.3f}', 
                        ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return shap_grid
```

Check Group 1 model names before creating the explainers


```python
# Check Group 1 models
print("GROUP 1 MODELS:")
print("=" * 50)
print(f"Available models: {list(trainer_group1.models.keys())}")
print(f"Available test data: {list(trainer_group1.test_data.keys())}")
```

    GROUP 1 MODELS:
    ==================================================
    Available models: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
    Available test data: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
    

Implement CustomSHAPExplainer to create SHAP explainers for Group 1 (8 features)


```python
test_data_group1 = trainer_group1.get_test_data('smooth_gradient')
X_test_group1 = test_data_group1['X_test']
y_test_group1 = test_data_group1['y_test']

shap_explainer = CustomSHAPExplainer(
    trainer=trainer_group1,
    dataset_name='smooth_gradient',  # Same name for both groups
    X_data=X_test_group1,
    roi_labels=None,
    coords=None,
    grid_size=None,
    background_type='mean',
    mode='decision_aligned'
)

print("SHAP explainer created (Group 1)")
print(f"  Model: smooth_gradient")
print(f"  Features: {shap_explainer.n_features}")  # Should be 8
print(f"  Samples: {len(X_test_group1)}")
```

      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 8, Classes: 3
    SHAP explainer created (Group 1)
      Model: smooth_gradient
      Features: 8
      Samples: 2000
    

Implement CustomSHAPExplainer to create SHAP explainers for Group 2 (10 features)
Note: GeoSHAP Explainers are created further down.


```python
shap_explainer_group2 = CustomSHAPExplainer(
    trainer=trainer_group2,
    dataset_name='smooth_gradient',
    X_data=X_test_group2,
    roi_labels=None,
    coords=None,
    grid_size=None,
    background_type='mean',
    mode='decision_aligned'
)

print("Standard SHAP explainer created (Group 2)")
print(f"  Features: {shap_explainer_group2.n_features}")  # Should be 10
print(f"  Treats all features independently (including x, y)")
print()

```

      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 10, Classes: 3
    ✓ Standard SHAP explainer created (Group 2)
      Features: 10
      Treats all features independently (including x, y)
    
    

GeoSHAP Explainer Implementation</br>
Based on: Li, Z. (2024). GeoShapley: A Game Theory Approach to Measuring Spatial Effects 
in Machine Learning Models. Annals of the American Association of Geographers, 114(7), 1365-1385.

Key Difference from Standard SHAP:
- Spatial coordinates (x, y) are treated as a JOINT PLAYER
- They must always be included/excluded together in coalitions
- This reduces coalition space from 2^p to 2^(p-g+1) where g=2 for coordinates

Mathematical Framework:
1. φ_GEO: Location effect (Equation 7)
2. φ_j: Individual feature effects (Equation 8)  
3. φ_(GEO,j): Interaction effects between location and features (Equation 9)
4. Additivity: f(x) = f(background) + φ_GEO + Σφ_j + Σφ_(GEO,j) (Equation 10)


```python
"""
===================================================================================
DO NOT USE! DID NOT WORK!
===================================================================================
According to the article, the authors used a Kernel SHAP implementation

This version uses Kernel SHAP approximation (like the original geoshapley library)
instead of exact Shapley value calculation. But this version continues to have additivity issues.

Based on Li (2024) but using sampling-based Kernel SHAP.
"""

class KernelGeoSHAPExplainer:
    """
    GeoSHAP implementation using Kernel SHAP approximation
    
    Much faster than exact calculation and should have perfect additivity.
    """
    
    def __init__(self, trainer, dataset_name, X_data, spatial_feature_indices,
                 background_type='mean', mode='decision_aligned', 
                 roi_labels=None, coords=None, grid_size=None,
                 n_samples=2048):
        """
        Initialize Kernel GeoSHAP Explainer
        
        Parameters:
        -----------
        trainer : SimpleMLPModels instance
        dataset_name : str
        X_data : np.ndarray
            Feature data (n_samples, n_features)
        spatial_feature_indices : list
            Indices of spatial features (e.g., [8, 9])
        background_type : str
            'mean' or 'queen'
        mode : str
            'decision_aligned' or 'class_specific'
        roi_labels : np.ndarray, optional
        coords : np.ndarray, optional
        grid_size : int, optional
        n_samples : int
            Number of samples for Kernel SHAP (default: 2048)
        """
        self.trainer = trainer
        self.dataset_name = dataset_name
        self.model = trainer.get_model(dataset_name)
        
        if self.model is None:
            raise ValueError(f"No trained model found for dataset: {dataset_name}")
        
        self.X_data = X_data
        self.n_samples_data = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_classes = self.model.output_shape[1]
        
        # Spatial configuration
        self.spatial_feature_indices = list(spatial_feature_indices)
        self.n_spatial_features = len(self.spatial_feature_indices)
        self.n_nonspatial_features = self.n_features - self.n_spatial_features
        
        self.nonspatial_feature_indices = [
            i for i in range(self.n_features) 
            if i not in self.spatial_feature_indices
        ]
        
        self.background_type = background_type
        self.mode = mode
        self.n_samples = n_samples  # Kernel SHAP samples
        
        # Spatial information
        self.roi_labels = roi_labels
        self.coords = coords
        self.grid_size = grid_size
        
        # Compute backgrounds
        self.backgrounds = self._compute_backgrounds()
        
        self._print_initialization_summary()
    
    def _print_initialization_summary(self):
        """Print initialization summary"""
        print(f"\n{'='*70}")
        print(f"Kernel GeoSHAP Explainer Initialized: {self.dataset_name}")
        print(f"{'='*70}")
        print(f"Data: {self.n_samples_data} samples, {self.n_features} features")
        print(f"Spatial features (joint): {self.spatial_feature_indices}")
        print(f"Non-spatial features: {self.nonspatial_feature_indices}")
        print(f"Configuration: {self.background_type} background, {self.mode} mode")
        print(f"Kernel SHAP samples: {self.n_samples}")
        print(f"{'='*70}\n")
    
    def _compute_backgrounds(self):
        """Compute background values"""
        backgrounds = {}
        
        if self.background_type == 'mean':
            global_mean = np.mean(self.X_data, axis=0)
            if self.roi_labels is not None:
                for roi_id in np.unique(self.roi_labels):
                    backgrounds[roi_id] = global_mean
            else:
                for i in range(len(self.X_data)):
                    backgrounds[i] = global_mean
                    
        elif self.background_type == 'queen':
            if self.roi_labels is None or self.coords is None or self.grid_size is None:
                raise ValueError("roi_labels, coords, and grid_size required for queen background")
            
            for roi_id in np.unique(self.roi_labels):
                queen_samples = self._get_queen_neighbors(roi_id)
                if len(queen_samples) > 0:
                    backgrounds[roi_id] = np.mean(queen_samples, axis=0)
                else:
                    backgrounds[roi_id] = np.mean(self.X_data, axis=0)
        
        return backgrounds
    
    def _get_queen_neighbors(self, roi_id):
        """Get samples from 8 surrounding neighbors"""
        row, col = self.coords[roi_id]
        neighbor_samples = []
        
        directions = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
        
        for dr, dc in directions:
            neighbor_row, neighbor_col = row + dr, col + dc
            
            if 0 <= neighbor_row < self.grid_size and 0 <= neighbor_col < self.grid_size:
                neighbor_roi_id = neighbor_row * self.grid_size + neighbor_col
                neighbor_mask = self.roi_labels == neighbor_roi_id
                neighbor_data = self.X_data[neighbor_mask]
                
                if len(neighbor_data) > 0:
                    neighbor_samples.append(neighbor_data)
        
        if neighbor_samples:
            return np.vstack(neighbor_samples)
        else:
            return np.array([])
    
    def _get_background_for_sample(self, sample_idx):
        """Get background for a sample"""
        if self.roi_labels is not None:
            roi_id = self.roi_labels[sample_idx]
            return self.backgrounds[roi_id]
        else:
            return self.backgrounds[sample_idx]
    
    def _get_model_output(self, X, output_type='logits'):
        """Get model output"""
        if output_type == 'logits':
            return self.trainer.get_logits_prediction(self.dataset_name, X, verbose=0)
        else:
            return self.trainer.get_probabilities_prediction(self.dataset_name, X, verbose=0)
    
    def _kernel_shap_sample_coalitions(self):
        """
        Generate coalition samples for Kernel SHAP
        
        Uses weighted sampling to focus on important coalition sizes
        """
        np.random.seed(42)
        
        # Total number of players in GeoSHAP framework
        # GEO counts as 1 player, plus n_nonspatial features
        n_players = self.n_nonspatial_features + 1
        
        coalitions = []
        
        # Always include empty and full coalitions
        coalitions.append({'geo': False, 'features': []})
        coalitions.append({'geo': True, 'features': list(self.nonspatial_feature_indices)})
        
        # Sample random coalitions
        for _ in range(self.n_samples - 2):
            # Decide if GEO is included (50% probability)
            include_geo = np.random.rand() < 0.5
            
            # Decide how many non-spatial features to include
            n_features_to_include = np.random.randint(0, self.n_nonspatial_features + 1)
            
            # Randomly select which features
            if n_features_to_include > 0:
                selected_features = list(np.random.choice(
                    self.nonspatial_feature_indices, 
                    size=n_features_to_include, 
                    replace=False
                ))
            else:
                selected_features = []
            
            coalitions.append({'geo': include_geo, 'features': selected_features})
        
        return coalitions
    
    def _evaluate_coalition(self, sample, background, coalition, target_class):
        """
        Evaluate model for a specific coalition
        
        Returns model output for input where:
        - Features in coalition use sample values
        - Features not in coalition use background values
        """
        model_input = background.copy()
        
        # Add GEO if included
        if coalition['geo']:
            model_input[self.spatial_feature_indices] = sample[self.spatial_feature_indices]
        
        # Add selected features
        for feat_idx in coalition['features']:
            model_input[feat_idx] = sample[feat_idx]
        
        output = self._get_model_output(model_input.reshape(1, -1), output_type='logits')
        return output[0, target_class]
    
    def explain_sample(self, sample_idx, target_class=None):
        """
        Compute GeoSHAP values for a single sample using Kernel SHAP
        
        Returns:
        --------
        dict with GeoSHAP decomposition
        """
        sample = self.X_data[sample_idx]
        background = self._get_background_for_sample(sample_idx)
        
        # Auto-detect target class
        if self.mode == 'decision_aligned' and target_class is None:
            sample_output = self._get_model_output(sample.reshape(1, -1), output_type='logits')
            target_class = np.argmax(sample_output[0])
        
        # Get baseline and full value
        baseline_output = self._get_model_output(background.reshape(1, -1), output_type='logits')
        baseline = baseline_output[0, target_class]
        
        full_output = self._get_model_output(sample.reshape(1, -1), output_type='logits')
        full_value = full_output[0, target_class]
        
        # Generate coalitions
        coalitions = self._kernel_shap_sample_coalitions()
        
        # Evaluate all coalitions
        coalition_values = np.array([
            self._evaluate_coalition(sample, background, c, target_class)
            for c in coalitions
        ])
        
        # Use linear regression to solve for Shapley values
        # This is the Kernel SHAP approach
        from sklearn.linear_model import LinearRegression
        
        # Create design matrix (which features are present in each coalition)
        n_players = self.n_nonspatial_features + 1  # +1 for GEO
        X_design = np.zeros((len(coalitions), n_players))
        
        for i, coalition in enumerate(coalitions):
            # GEO is player 0
            if coalition['geo']:
                X_design[i, 0] = 1
            
            # Non-spatial features are players 1 to n_nonspatial_features
            for feat_idx in coalition['features']:
                player_idx = self.nonspatial_feature_indices.index(feat_idx) + 1
                X_design[i, player_idx] = 1
        
        # Fit linear model: f(coalition) = baseline + Σ(player_i * phi_i)
        # Target is: coalition_value - baseline
        y_target = coalition_values - baseline
        
        # Fit without intercept since we already subtracted baseline
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X_design, y_target)
        
        # Extract Shapley values
        shapley_values = reg.coef_
        
        phi_GEO = shapley_values[0]
        phi_features = shapley_values[1:]
        
        # For now, set interactions to zero (pure additive model)
        # The original GeoSHAP paper computes interactions separately
        phi_interactions = np.zeros(self.n_nonspatial_features)
        
        # Verify additivity
        predicted_full = baseline + phi_GEO + np.sum(phi_features) + np.sum(phi_interactions)
        additivity_error = abs(predicted_full - full_value)
        
        return {
            'phi_GEO': phi_GEO,
            'phi_features': phi_features,
            'phi_interactions': phi_interactions,
            'baseline': baseline,
            'full_value': full_value,
            'predicted_full': predicted_full,
            'additivity_error': additivity_error,
            'target_class': target_class,
            'nonspatial_feature_indices': self.nonspatial_feature_indices
        }
    
    def explain_samples(self, sample_indices, target_class=None):
        """Compute GeoSHAP values for multiple samples"""
        print(f"\nComputing Kernel GeoSHAP for {len(sample_indices)} samples...")
        
        all_results = []
        
        for sample_idx in tqdm(sample_indices, desc="Computing Kernel GeoSHAP"):
            if self.mode == 'decision_aligned' and target_class is None:
                result = self.explain_sample(sample_idx)
            else:
                result = self.explain_sample(sample_idx, target_class=target_class)
            
            all_results.append(result)
        
        # Aggregate results
        phi_GEO_values = np.array([r['phi_GEO'] for r in all_results])
        phi_features_values = np.array([r['phi_features'] for r in all_results])
        phi_interactions_values = np.array([r['phi_interactions'] for r in all_results])
        
        additivity_errors = [r['additivity_error'] for r in all_results]
        
        print(f"\nAdditivity Check:")
        print(f"  Mean error: {np.mean(additivity_errors):.6e}")
        print(f"  Max error: {np.max(additivity_errors):.6e}")
        
        return {
            'phi_GEO_values': phi_GEO_values,
            'phi_features_values': phi_features_values,
            'phi_interactions_values': phi_interactions_values,
            'individual_results': all_results,
            'mean_additivity_error': np.mean(additivity_errors),
            'max_additivity_error': np.max(additivity_errors),
            'configuration': (self.background_type, self.mode),
            'nonspatial_feature_indices': self.nonspatial_feature_indices
        }
```


```python
# Test it
from geoshapley import GeoShapleyExplainer

# Create wrapper for your model
def model_predict(X):
    return trainer_group2.get_probabilities_prediction('smooth_gradient', X, verbose=0)

# Sample background
background = X_test_group2[:100]  # Use 100 samples

# Initialize
explainer = GeoShapleyExplainer(
    model_predict,
    background,
    n_geo_features=2  # x, y coordinates
)

# Explain samples
result = explainer.explain(X_test_group2[0:10])

# Check results
print(f"Primary effects shape: {result.primary_effects.shape}")
print(f"GEO effects shape: {result.geo_effects.shape}")
print(f"Interaction effects shape: {result.interaction_effects.shape}")
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[183], line 12
          9 background = X_test_group2[:100]  # Use 100 samples
         11 # Initialize
    ---> 12 explainer = GeoShapleyExplainer(
         13     model_predict,
         14     background,
         15     n_geo_features=2  # x, y coordinates
         16 )
         18 # Explain samples
         19 result = explainer.explain(X_test_group2[0:10])
    

    TypeError: GeoShapleyExplainer.__init__() got an unexpected keyword argument 'n_geo_features'



```python
# Now re-create the explainer
test_data_group2 = trainer_group2.get_test_data('smooth_gradient')
X_test_group2 = test_data_group2['X_test']

geoshap_test = GeoSHAPExplainer(
    trainer=trainer_group2,
    dataset_name='smooth_gradient',
    X_data=X_test_group2,
    spatial_feature_indices=[8, 9],
    background_type='mean',
    mode='decision_aligned',
    roi_labels=None,
    coords=None,
    grid_size=None
)

# Test one sample
sample_idx = 0
result = geoshap_test.explain_sample(sample_idx)

print("\nGEOSHAP ADDITIVITY TEST (AFTER FIX)")
print("=" * 60)
print(f"Sample {sample_idx}:")
print(f"  Error: {result['additivity_error']:.6e}")
print(f"  Expected: < 1e-6")

if result['additivity_error'] < 1e-6:
    print(f"  ✓✓✓ ADDITIVITY FIXED! ✓✓✓")
else:
    print(f"  ✗ Still has errors - need to investigate further")
```

    
    ======================================================================
    GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data Configuration:
      Samples: 2,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
    
    GEOSHAP ADDITIVITY TEST (AFTER FIX)
    ============================================================
    Sample 0:
      Error: 4.200899e-02
      Expected: < 1e-6
      ✗ Still has errors - need to investigate further
    


```python
# Create explainer
geoshap = KernelGeoSHAPExplainer(
    trainer=trainer_group2,
    dataset_name='smooth_gradient',
    X_data=X_test_group2,
    spatial_feature_indices=[8, 9],
    background_type='mean',
    mode='decision_aligned',
    roi_labels=None,
    coords=None,
    grid_size=None,
    n_samples=2048  # Number of Kernel SHAP samples
)

# Test on one sample
result = geoshap.explain_sample(0)

print(f"\nKernel GeoSHAP Results:")
print(f"  φ_GEO: {result['phi_GEO']:.6f}")
print(f"  Σφ_features: {np.sum(result['phi_features']):.6f}")
print(f"  Baseline: {result['baseline']:.6f}")
print(f"  Full value: {result['full_value']:.6f}")
print(f"  Predicted: {result['predicted_full']:.6f}")
print(f"  Additivity error: {result['additivity_error']:.6e}")
```

    
    ======================================================================
    Kernel GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data: 2000 samples, 10 features
    Spatial features (joint): [8, 9]
    Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7]
    Configuration: mean background, decision_aligned mode
    Kernel SHAP samples: 2048
    ======================================================================
    
    
    Kernel GeoSHAP Results:
      φ_GEO: 0.722152
      Σφ_features: 13.314705
      Baseline: -4.794620
      Full value: 9.273224
      Predicted: 9.242237
      Additivity error: 3.098723e-02
    


```python
# Create a standard SHAP explainer for comparison
# For mean background with no ROI info, we need to pass sample-level backgrounds
shap_test = CustomSHAPExplainer(
    trainer=trainer_group2,
    dataset_name='smooth_gradient',
    X_data=X_test_group2,
    roi_labels=np.arange(len(X_test_group2)),  # Treat each sample as its own "ROI"
    coords=None,
    grid_size=None,
    background_type='mean',
    mode='decision_aligned'
)

shap_result = shap_test.explain_sample(sample_idx)

print(f"\nStandard SHAP:")
print(f"  Baseline: {shap_result['baseline']:.6f}")
print(f"  Full value: {shap_result['full_value']:.6f}")
print(f"  Sum of SHAP values: {np.sum(shap_result['shap_values']):.6f}")
print(f"  Expected (full - baseline): {shap_result['full_value'] - shap_result['baseline']:.6f}")
print(f"  Additivity error: {shap_result['additivity_error']:.6e}")

print(f"\nStandard SHAP values for all 10 features:")
for i in range(10):
    print(f"  Feature {i}: {shap_result['shap_values'][i]:.6f}")

print(f"\n  φ_x + φ_y (spatial features): {shap_result['shap_values'][8] + shap_result['shap_values'][9]:.6f}")
print(f"  GeoSHAP φ_GEO: {result['phi_GEO']:.6f}")
print(f"  Difference: {abs((shap_result['shap_values'][8] + shap_result['shap_values'][9]) - result['phi_GEO']):.6f}")

print(f"\n  Sum of SHAP for features 0-7: {np.sum(shap_result['shap_values'][0:8]):.6f}")
print(f"  GeoSHAP Σφ_j + Σφ_(GEO,j): {np.sum(result['phi_features']) + np.sum(result['phi_interactions']):.6f}")
print(f"  Difference: {abs(np.sum(shap_result['shap_values'][0:8]) - (np.sum(result['phi_features']) + np.sum(result['phi_interactions']))):.6f}")
```

      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 10, Classes: 3
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
    
    Standard SHAP:
      Baseline: -4.794620
      Full value: 9.273224
      Sum of SHAP values: 14.067845
      Expected (full - baseline): 14.067844
      Additivity error: 8.344650e-07
    
    Standard SHAP values for all 10 features:
      Feature 0: 3.927728
      Feature 1: 2.391431
      Feature 2: 1.021878
      Feature 3: 3.917814
      Feature 4: 1.483168
      Feature 5: 0.821502
      Feature 6: -0.556832
      Feature 7: 0.257363
      Feature 8: -0.500849
      Feature 9: 1.304642
    
      φ_x + φ_y (spatial features): 0.803793
      GeoSHAP φ_GEO: 0.791789
      Difference: 0.012004
    
      Sum of SHAP for features 0-7: 13.264052
      GeoSHAP Σφ_j + Σφ_(GEO,j): 13.318063
      Difference: 0.054012
    

Check Group 2 model names before creating the explainers


```python
# Check Group 2 models
print("\nGROUP 2 MODELS:")
print("=" * 50)
print(f"Available models: {list(trainer_group2.models.keys())}")
print(f"Available test data: {list(trainer_group2.test_data.keys())}")
```

    
    GROUP 2 MODELS:
    ==================================================
    Available models: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
    Available test data: ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']
    

Create GeoSHAP Explainer for Group 2 (10 features)


```python
test_data_group2 = trainer_group2.get_test_data('smooth_gradient')
X_test_group2 = test_data_group2['X_test']
y_test_group2 = test_data_group2['y_test']

geoshap_explainer = GeoSHAPExplainer(
    trainer=trainer_group2,
    dataset_name='smooth_gradient',  # Same name, different trainer
    X_data=X_test_group2,
    spatial_feature_indices=[8, 9],  # Last 2 features are x, y
    background_type='mean',
    mode='decision_aligned',
    roi_labels=None,
    coords=None,
    grid_size=None
)

print("\nGeoSHAP explainer created (Group 2)")
print(f"  Model: smooth_gradient")
print(f"  Total features: {geoshap_explainer.n_features}")  # Should be 10
print(f"  Spatial features: {geoshap_explainer.spatial_feature_indices}")
print(f"  Non-spatial features: {geoshap_explainer.nonspatial_feature_indices}")
print(f"  Samples: {len(X_test_group2)}")
```

    
    ======================================================================
    GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data Configuration:
      Samples: 2,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    
    GeoSHAP explainer created (Group 2)
      Model: smooth_gradient
      Total features: 10
      Spatial features: [8, 9]
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7]
      Samples: 2000
    

## Comprehensive Test

1. comprehensive_shap_test - Testing Phase

    Purpose: Quick validation that SHAP works correctly
    What it does:

    Tests only ~10-13 samples per configuration
    Verifies additivity (SHAP values sum correctly)
    Tests basic functionality


    Output: Stores the explainer objects for later use
    When to use: Once per dataset to verify configurations work

2. test_configuration_differences - Comparison Phase

    Purpose: Compare the configurations from the test
    What it does:

    Uses the few samples from comprehensive_shap_test
    Compares SHAP values across configurations
    Shows correlation, RMSE, differences


    Output: Statistical comparison metrics
    When to use: After comprehensive_shap_test to decide which configurations to use

3. run_full_shap_analysis_on_selected_samples - Full Analysis Phase

    Purpose: Get SHAP values for ALL selected samples (100 samples, 1 per ROI)
    What it does:

    Takes an explainer (from comprehensive_shap_test)
    Runs SHAP on all 100 selected samples
    Returns organized results for downstream analysis (DCG, visualization)


    Output: Complete SHAP values for spatial analysis
    When to use: After deciding which configuration(s) to use, to get full results

comprehensive_shap_test → test_configuration_differences → [decide configs] → run_full_shap_analysis_on_selected_samples → DCG Analysis</br>
&emsp;&emsp;&emsp;&emsp;↓&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;↓</br>
&emsp;&emsp;&emsp;(Test 10 samples)&emsp;&emsp;&emsp;&emsp;(Compare configs)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(Analyze 100 samples)

## Test for additivity


```python
def test_explainer_additivity(trainer, analysis_data, dataset_name='smooth_gradient', 
                               explainer_type='shap'):
    """
    Test additivity and basic functionality for SHAP or GeoSHAP explainer
    
    Parameters:
    -----------
    trainer : SimpleMLPModels instance
        Trained model wrapper
    analysis_data : dict
        Results from setup_consistent_samples_for_analysis
        Must contain dataset_dict with X, roi_labels, coords, grid_size
    dataset_name : str
        Which dataset to test (e.g., 'smooth_gradient')
    explainer_type : str
        'shap' or 'geoshap'
    
    Returns:
    --------
    test_results : dict
        Dictionary with results from all 4 configurations
    """
    print(f"=== ADDITIVITY TEST FOR {explainer_type.upper()} - {dataset_name.upper()} ===")
    print("=" * 60)
    
    # Get the consistent samples for this dataset
    dataset_info = analysis_data[dataset_name]
    selected_samples = dataset_info['selected_samples']
    dataset_dict = dataset_info['dataset']
    
    # Check if this is Group 2 data (for GeoSHAP)
    n_features = dataset_dict['X'].shape[1]
    is_group2 = (n_features == 10)
    
    if explainer_type == 'geoshap' and not is_group2:
        raise ValueError(f"GeoSHAP requires Group 2 data (10 features), but got {n_features} features")
    
    # Test configurations: [background_type, mode]
    configurations = [
        ('mean', 'decision_aligned'),      # Global + Decision-aligned
        ('queen', 'decision_aligned'),     # Queen + Decision-aligned  
        ('mean', 'class_specific'),        # Global + Class-specific
        ('queen', 'class_specific')        # Queen + Class-specific
    ]
    
    test_results = {}
    
    for config_idx, (bg_type, mode) in enumerate(configurations, 1):
        config_name = f"{bg_type}_{mode}"
        print(f"\n--- Configuration {config_idx}: {bg_type.title()} Background + "
              f"{mode.replace('_', ' ').title()} ---")
        
        try:
            # Initialize explainer based on type
            if explainer_type == 'shap':
                explainer = CustomSHAPExplainer(
                    trainer=trainer,
                    dataset_name=dataset_name,
                    X_data=dataset_dict['X'],
                    roi_labels=dataset_dict['roi_labels'], 
                    coords=dataset_dict['coords'],
                    grid_size=dataset_dict['grid_size'],
                    background_type=bg_type,
                    mode=mode
                )
            elif explainer_type == 'geoshap':
                explainer = GeoSHAPExplainer(
                    trainer=trainer,
                    dataset_name=dataset_name,
                    X_data=dataset_dict['X'],
                    spatial_feature_indices=[8, 9],
                    background_type=bg_type,
                    mode=mode,
                    roi_labels=dataset_dict['roi_labels'],
                    coords=dataset_dict['coords'],
                    grid_size=dataset_dict['grid_size']
                )
            else:
                raise ValueError(f"Unknown explainer_type: {explainer_type}")
            
            # Test basic functionality
            print(f"Testing basic functionality...")
            
            # Get first selected sample for testing
            first_sample_idx = selected_samples['sample_indices'][0]
            
            # For class-specific mode, test with each class
            if mode == 'class_specific':
                print(f"Testing class-specific mode with all classes...")
                class_results = []
                
                for target_class in range(explainer.n_classes):
                    result = explainer.explain_sample(first_sample_idx, target_class=target_class)
                    class_results.append(result)
                    print(f"  Class {target_class}: Baseline={result['baseline']:.4f}, "
                          f"Additivity Error={result['additivity_error']:.2e}")
                
                # Use class 0 for subsequent tests
                test_result = class_results[0]
            else:
                # Decision-aligned mode
                test_result = explainer.explain_sample(first_sample_idx)
                print(f"  Auto-detected class: {test_result['target_class']}")
                print(f"  Baseline: {test_result['baseline']:.4f}")
                print(f"  Additivity error: {test_result['additivity_error']:.2e}")
            
            # Test additivity across multiple samples
            print(f"Testing additivity across 10 samples...")
            additivity_errors = []
            
            for i in range(min(10, len(selected_samples['sample_indices']))):
                sample_idx = selected_samples['sample_indices'][i]
                
                if mode == 'class_specific':
                    # Use predicted class for consistency
                    sample_pred = trainer.get_probabilities_prediction(
                        dataset_name, 
                        dataset_dict['X'][sample_idx:sample_idx+1]
                    )
                    pred_class = np.argmax(sample_pred[0])
                    result = explainer.explain_sample(sample_idx, target_class=pred_class)
                else:
                    result = explainer.explain_sample(sample_idx)
                
                additivity_errors.append(result['additivity_error'])
            
            print(f"  Mean additivity error: {np.mean(additivity_errors):.2e}")
            print(f"  Max additivity error: {np.max(additivity_errors):.2e}")
            
            # Check if additivity passed
            additivity_passed = np.all(np.array(additivity_errors) < 1e-6)
            if additivity_passed:
                print(f"  ✓ Additivity check PASSED")
            else:
                print(f"  ⚠ Warning: Some additivity errors exceed 1e-6")
            
            # Store results
            test_results[config_name] = {
                'configuration': (bg_type, mode),
                'explainer': explainer,
                'basic_test_result': test_result,
                'additivity_errors': additivity_errors,
                'mean_additivity_error': np.mean(additivity_errors),
                'max_additivity_error': np.max(additivity_errors),
                'additivity_passed': additivity_passed,
                'test_status': 'PASSED'
            }
            
            print(f"✓ Configuration {config_name} PASSED")
            
        except Exception as e:
            print(f"✗ Configuration {config_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            
            test_results[config_name] = {
                'configuration': (bg_type, mode),
                'test_status': 'FAILED',
                'error': str(e)
            }
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"TEST SUMMARY: {explainer_type.upper()} - {dataset_name.upper()}")
    print(f"=" * 60)
    
    passed_configs = [name for name, result in test_results.items() 
                     if result.get('test_status') == 'PASSED']
    failed_configs = [name for name, result in test_results.items() 
                     if result.get('test_status') == 'FAILED']
    
    print(f"Configurations tested: {len(test_results)}")
    print(f"Passed: {len(passed_configs)} - {passed_configs}")
    print(f"Failed: {len(failed_configs)} - {failed_configs}")
    
    if passed_configs:
        print(f"\nAdditivity Check Summary:")
        for config_name in passed_configs:
            result = test_results[config_name]
            status = "✓ PASS" if result['additivity_passed'] else "⚠ WARNING"
            print(f"  {config_name}: {status} - Mean={result['mean_additivity_error']:.2e}, "
                  f"Max={result['max_additivity_error']:.2e}")
    
    return test_results

```


```python
# Test SHAP additivity on Group 1 data
shap_test_results = test_explainer_additivity(
    trainer=trainer_group1,
    analysis_data=analysis_data_group1,
    dataset_name='smooth_gradient',
    explainer_type='shap'
)
```

    === ADDITIVITY TEST FOR SHAP - SMOOTH_GRADIENT ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: -5.9359
      Additivity error: 1.43e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.39e-06
      Max additivity error: 3.20e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: 24.2444
      Additivity error: 2.57e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.11e-06
      Max additivity error: 5.58e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-9.9147, Additivity Error=1.14e-05
      Class 1: Baseline=4.5595, Additivity Error=2.98e-07
      Class 2: Baseline=-5.9359, Additivity Error=1.43e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.39e-06
      Max additivity error: 3.20e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-55.8512, Additivity Error=9.51e-06
      Class 1: Baseline=-2.7355, Additivity Error=7.02e-07
      Class 2: Baseline=24.2444, Additivity Error=2.57e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.11e-06
      Max additivity error: 5.58e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - SMOOTH_GRADIENT
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=1.39e-06, Max=3.20e-06
      queen_decision_aligned: ⚠ WARNING - Mean=1.11e-06, Max=5.58e-06
      mean_class_specific: ⚠ WARNING - Mean=1.39e-06, Max=3.20e-06
      queen_class_specific: ⚠ WARNING - Mean=1.11e-06, Max=5.58e-06
    

Testing SHAP additivity on all group 1 datasets

Note: For convenience, I put in a test status "Passed". The break point for passed, 1e-6, is arbitrary. You might want to use a stricter or less strict break point. Also, if you keep my break point, you might want to check if the additivity is reasonable even if it does not "Pass".


```python

```


```python
dataset_names = ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']

shap_all_results = {}

for dataset_name in dataset_names:
    print(f"\n{'='*70}")
    print(f"Testing {dataset_name}...")
    print(f"{'='*70}")
    
    results = test_explainer_additivity(
        trainer=trainer_group1,
        analysis_data=analysis_data_group1,
        dataset_name=dataset_name,
        explainer_type='shap'
    )
    
    shap_all_results[dataset_name] = results

# Summary for all datasets
print("\n" + "=" * 70)
print("SHAP ADDITIVITY SUMMARY - ALL DATASETS")
print("=" * 70)

for dataset_name, results in shap_all_results.items():
    passed = sum(1 for r in results.values() if r.get('test_status') == 'PASSED')
    total = len(results)
    print(f"\n{dataset_name}: {passed}/{total} configurations passed")
    
    for config_name, result in results.items():
        if result.get('test_status') == 'PASSED':
            status = "✓" if result['additivity_passed'] else "⚠"
            print(f"  {status} {config_name}: mean={result['mean_additivity_error']:.2e}, "
                  f"max={result['max_additivity_error']:.2e}")
```

    
    ======================================================================
    Testing smooth_gradient...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - SMOOTH_GRADIENT ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: -5.9359
      Additivity error: 1.43e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.39e-06
      Max additivity error: 3.20e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: 24.2444
      Additivity error: 2.57e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.11e-06
      Max additivity error: 5.58e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-9.9147, Additivity Error=1.14e-05
      Class 1: Baseline=4.5595, Additivity Error=2.98e-07
      Class 2: Baseline=-5.9359, Additivity Error=1.43e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.39e-06
      Max additivity error: 3.20e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-55.8512, Additivity Error=9.51e-06
      Class 1: Baseline=-2.7355, Additivity Error=7.02e-07
      Class 2: Baseline=24.2444, Additivity Error=2.57e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.11e-06
      Max additivity error: 5.58e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - SMOOTH_GRADIENT
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=1.39e-06, Max=3.20e-06
      queen_decision_aligned: ⚠ WARNING - Mean=1.11e-06, Max=5.58e-06
      mean_class_specific: ⚠ WARNING - Mean=1.39e-06, Max=3.20e-06
      queen_class_specific: ⚠ WARNING - Mean=1.11e-06, Max=5.58e-06
    
    ======================================================================
    Testing sharp_boundary...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - SHARP_BOUNDARY ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: mean
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: 0.8427
      Additivity error: 4.47e-08
    Testing additivity across 10 samples...
      Mean additivity error: 3.96e-07
      Max additivity error: 1.30e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: queen
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: 2.8123
      Additivity error: 8.34e-07
    Testing additivity across 10 samples...
      Mean additivity error: 4.61e-07
      Max additivity error: 1.35e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: mean
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=1.2649, Additivity Error=1.01e-06
      Class 1: Baseline=2.7766, Additivity Error=6.03e-07
      Class 2: Baseline=0.8427, Additivity Error=4.47e-08
    Testing additivity across 10 samples...
      Mean additivity error: 3.96e-07
      Max additivity error: 1.30e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: queen
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-1.7466, Additivity Error=7.71e-07
      Class 1: Baseline=2.4113, Additivity Error=7.35e-08
      Class 2: Baseline=2.8123, Additivity Error=8.34e-07
    Testing additivity across 10 samples...
      Mean additivity error: 4.61e-07
      Max additivity error: 1.35e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - SHARP_BOUNDARY
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=3.96e-07, Max=1.30e-06
      queen_decision_aligned: ⚠ WARNING - Mean=4.61e-07, Max=1.35e-06
      mean_class_specific: ⚠ WARNING - Mean=3.96e-07, Max=1.30e-06
      queen_class_specific: ⚠ WARNING - Mean=4.61e-07, Max=1.35e-06
    
    ======================================================================
    Testing clustered_hotspots...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - CLUSTERED_HOTSPOTS ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: mean
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 0
      Baseline: -0.2034
      Additivity error: 2.61e-08
    Testing additivity across 10 samples...
      Mean additivity error: 6.71e-08
      Max additivity error: 2.09e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: queen
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 0
      Baseline: -1.6876
      Additivity error: 8.94e-08
    Testing additivity across 10 samples...
      Mean additivity error: 7.86e-08
      Max additivity error: 2.12e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: mean
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-0.2034, Additivity Error=2.61e-08
      Class 1: Baseline=-0.4407, Additivity Error=4.47e-08
      Class 2: Baseline=0.4215, Additivity Error=2.61e-08
    Testing additivity across 10 samples...
      Mean additivity error: 6.71e-08
      Max additivity error: 2.09e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: queen
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-1.6876, Additivity Error=8.94e-08
      Class 1: Baseline=-0.2853, Additivity Error=7.08e-08
      Class 2: Baseline=-0.3107, Additivity Error=1.52e-07
    Testing additivity across 10 samples...
      Mean additivity error: 7.86e-08
      Max additivity error: 2.12e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - CLUSTERED_HOTSPOTS
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ✓ PASS - Mean=6.71e-08, Max=2.09e-07
      queen_decision_aligned: ✓ PASS - Mean=7.86e-08, Max=2.12e-07
      mean_class_specific: ✓ PASS - Mean=6.71e-08, Max=2.09e-07
      queen_class_specific: ✓ PASS - Mean=7.86e-08, Max=2.12e-07
    
    ======================================================================
    Testing random_pattern...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - RANDOM_PATTERN ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: mean
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 0
      Baseline: 0.4127
      Additivity error: 3.87e-07
    Testing additivity across 10 samples...
      Mean additivity error: 1.79e-07
      Max additivity error: 3.87e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: queen
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 0
      Baseline: 0.5162
      Additivity error: 2.98e-07
    Testing additivity across 10 samples...
      Mean additivity error: 1.94e-07
      Max additivity error: 4.99e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: mean
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=0.4127, Additivity Error=3.87e-07
      Class 1: Baseline=-0.3308, Additivity Error=9.69e-08
      Class 2: Baseline=-0.4761, Additivity Error=2.46e-07
    Testing additivity across 10 samples...
      Mean additivity error: 1.79e-07
      Max additivity error: 3.87e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: queen
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=0.5162, Additivity Error=2.98e-07
      Class 1: Baseline=-0.5912, Additivity Error=8.94e-08
      Class 2: Baseline=-0.7199, Additivity Error=3.28e-07
    Testing additivity across 10 samples...
      Mean additivity error: 1.94e-07
      Max additivity error: 4.99e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - RANDOM_PATTERN
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ✓ PASS - Mean=1.79e-07, Max=3.87e-07
      queen_decision_aligned: ✓ PASS - Mean=1.94e-07, Max=4.99e-07
      mean_class_specific: ✓ PASS - Mean=1.79e-07, Max=3.87e-07
      queen_class_specific: ✓ PASS - Mean=1.94e-07, Max=4.99e-07
    
    ======================================================================
    SHAP ADDITIVITY SUMMARY - ALL DATASETS
    ======================================================================
    
    smooth_gradient: 4/4 configurations passed
      ⚠ mean_decision_aligned: mean=1.39e-06, max=3.20e-06
      ⚠ queen_decision_aligned: mean=1.11e-06, max=5.58e-06
      ⚠ mean_class_specific: mean=1.39e-06, max=3.20e-06
      ⚠ queen_class_specific: mean=1.11e-06, max=5.58e-06
    
    sharp_boundary: 4/4 configurations passed
      ⚠ mean_decision_aligned: mean=3.96e-07, max=1.30e-06
      ⚠ queen_decision_aligned: mean=4.61e-07, max=1.35e-06
      ⚠ mean_class_specific: mean=3.96e-07, max=1.30e-06
      ⚠ queen_class_specific: mean=4.61e-07, max=1.35e-06
    
    clustered_hotspots: 4/4 configurations passed
      ✓ mean_decision_aligned: mean=6.71e-08, max=2.09e-07
      ✓ queen_decision_aligned: mean=7.86e-08, max=2.12e-07
      ✓ mean_class_specific: mean=6.71e-08, max=2.09e-07
      ✓ queen_class_specific: mean=7.86e-08, max=2.12e-07
    
    random_pattern: 4/4 configurations passed
      ✓ mean_decision_aligned: mean=1.79e-07, max=3.87e-07
      ✓ queen_decision_aligned: mean=1.94e-07, max=4.99e-07
      ✓ mean_class_specific: mean=1.79e-07, max=3.87e-07
      ✓ queen_class_specific: mean=1.94e-07, max=4.99e-07
    

Testing SHAP additivity on all group 2 datasets

Note: For convenience, I put in a test status "Passed". The break point for passed is arbitrary. You might want to use a stricter or less strict break point. Also, if you keep my break point, you might want to check if the additivity is reasonable even if it does not "Pass".


```python
# Test SHAP additivity on Group 2 data
dataset_names = ['smooth_gradient', 'sharp_boundary', 'clustered_hotspots', 'random_pattern']

shap_all_results = {}

for dataset_name in dataset_names:
    print(f"\n{'='*70}")
    print(f"Testing {dataset_name}...")
    print(f"{'='*70}")
    
    results = test_explainer_additivity(
        trainer=trainer_group2,
        analysis_data=analysis_data_group2,
        dataset_name=dataset_name,
        explainer_type='shap'
    )
    
    shap_all_results[dataset_name] = results

# Summary for all datasets
print("\n" + "=" * 70)
print("SHAP ADDITIVITY SUMMARY - ALL DATASETS")
print("=" * 70)

for dataset_name, results in shap_all_results.items():
    passed = sum(1 for r in results.values() if r.get('test_status') == 'PASSED')
    total = len(results)
    print(f"\n{dataset_name}: {passed}/{total} configurations passed")
    
    for config_name, result in results.items():
        if result.get('test_status') == 'PASSED':
            status = "✓" if result['additivity_passed'] else "⚠"
            print(f"  {status} {config_name}: mean={result['mean_additivity_error']:.2e}, "
                  f"max={result['max_additivity_error']:.2e}")
```

    
    ======================================================================
    Testing smooth_gradient...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - SMOOTH_GRADIENT ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 2
      Baseline: -4.6048
      Additivity error: 5.01e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.43e-06
      Max additivity error: 5.01e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 2
      Baseline: 25.8321
      Additivity error: 3.79e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.67e-06
      Max additivity error: 4.12e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=-6.5495, Additivity Error=5.72e-06
      Class 1: Baseline=3.9212, Additivity Error=8.94e-08
      Class 2: Baseline=-4.6048, Additivity Error=5.01e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.43e-06
      Max additivity error: 5.01e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=-32.4103, Additivity Error=4.19e-06
      Class 1: Baseline=-3.8854, Additivity Error=1.53e-07
      Class 2: Baseline=25.8321, Additivity Error=3.79e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.67e-06
      Max additivity error: 4.12e-06
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - SMOOTH_GRADIENT
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=1.43e-06, Max=5.01e-06
      queen_decision_aligned: ⚠ WARNING - Mean=1.67e-06, Max=4.12e-06
      mean_class_specific: ⚠ WARNING - Mean=1.43e-06, Max=5.01e-06
      queen_class_specific: ⚠ WARNING - Mean=1.67e-06, Max=4.12e-06
    
    ======================================================================
    Testing sharp_boundary...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - SHARP_BOUNDARY ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: mean
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 2
      Baseline: 1.1419
      Additivity error: 1.86e-08
    Testing additivity across 10 samples...
      Mean additivity error: 2.27e-07
      Max additivity error: 6.00e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: queen
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 2
      Baseline: 3.4806
      Additivity error: 3.48e-07
    Testing additivity across 10 samples...
      Mean additivity error: 3.79e-07
      Max additivity error: 8.01e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: mean
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=0.9024, Additivity Error=1.60e-07
      Class 1: Baseline=3.0175, Additivity Error=6.15e-08
      Class 2: Baseline=1.1419, Additivity Error=1.86e-08
    Testing additivity across 10 samples...
      Mean additivity error: 2.27e-07
      Max additivity error: 6.00e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for sharp_boundary
      Background type: queen
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=-0.3980, Additivity Error=5.10e-08
      Class 1: Baseline=2.9298, Additivity Error=2.95e-08
      Class 2: Baseline=3.4806, Additivity Error=3.48e-07
    Testing additivity across 10 samples...
      Mean additivity error: 3.79e-07
      Max additivity error: 8.01e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - SHARP_BOUNDARY
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ✓ PASS - Mean=2.27e-07, Max=6.00e-07
      queen_decision_aligned: ✓ PASS - Mean=3.79e-07, Max=8.01e-07
      mean_class_specific: ✓ PASS - Mean=2.27e-07, Max=6.00e-07
      queen_class_specific: ✓ PASS - Mean=3.79e-07, Max=8.01e-07
    
    ======================================================================
    Testing clustered_hotspots...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - CLUSTERED_HOTSPOTS ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: mean
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 0
      Baseline: -0.5952
      Additivity error: 1.47e-08
    Testing additivity across 10 samples...
      Mean additivity error: 1.58e-07
      Max additivity error: 3.80e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: queen
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 0
      Baseline: -1.4762
      Additivity error: 3.33e-07
    Testing additivity across 10 samples...
      Mean additivity error: 2.83e-07
      Max additivity error: 4.50e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: mean
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=-0.5952, Additivity Error=1.47e-08
      Class 1: Baseline=1.2438, Additivity Error=8.94e-08
      Class 2: Baseline=1.4651, Additivity Error=2.98e-08
    Testing additivity across 10 samples...
      Mean additivity error: 1.58e-07
      Max additivity error: 3.80e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for clustered_hotspots
      Background type: queen
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=-1.4762, Additivity Error=3.33e-07
      Class 1: Baseline=2.0045, Additivity Error=2.46e-07
      Class 2: Baseline=3.1238, Additivity Error=8.20e-08
    Testing additivity across 10 samples...
      Mean additivity error: 2.83e-07
      Max additivity error: 4.50e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - CLUSTERED_HOTSPOTS
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ✓ PASS - Mean=1.58e-07, Max=3.80e-07
      queen_decision_aligned: ✓ PASS - Mean=2.83e-07, Max=4.50e-07
      mean_class_specific: ✓ PASS - Mean=1.58e-07, Max=3.80e-07
      queen_class_specific: ✓ PASS - Mean=2.83e-07, Max=4.50e-07
    
    ======================================================================
    Testing random_pattern...
    ======================================================================
    === ADDITIVITY TEST FOR SHAP - RANDOM_PATTERN ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: mean
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 0
      Baseline: 0.9583
      Additivity error: 8.20e-08
    Testing additivity across 10 samples...
      Mean additivity error: 2.47e-07
      Max additivity error: 5.30e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: queen
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 0
      Baseline: 0.7148
      Additivity error: 4.84e-08
    Testing additivity across 10 samples...
      Mean additivity error: 2.23e-07
      Max additivity error: 5.36e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: mean
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=0.9583, Additivity Error=8.20e-08
      Class 1: Baseline=-0.6415, Additivity Error=3.28e-07
      Class 2: Baseline=-1.1390, Additivity Error=2.68e-07
    Testing additivity across 10 samples...
      Mean additivity error: 2.47e-07
      Max additivity error: 5.30e-07
      ✓ Additivity check PASSED
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for random_pattern
      Background type: queen
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=0.7148, Additivity Error=4.84e-08
      Class 1: Baseline=-0.5670, Additivity Error=6.05e-08
      Class 2: Baseline=-0.7054, Additivity Error=7.82e-08
    Testing additivity across 10 samples...
      Mean additivity error: 2.23e-07
      Max additivity error: 5.36e-07
      ✓ Additivity check PASSED
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: SHAP - RANDOM_PATTERN
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ✓ PASS - Mean=2.47e-07, Max=5.30e-07
      queen_decision_aligned: ✓ PASS - Mean=2.23e-07, Max=5.36e-07
      mean_class_specific: ✓ PASS - Mean=2.47e-07, Max=5.30e-07
      queen_class_specific: ✓ PASS - Mean=2.23e-07, Max=5.36e-07
    
    ======================================================================
    SHAP ADDITIVITY SUMMARY - ALL DATASETS
    ======================================================================
    
    smooth_gradient: 4/4 configurations passed
      ⚠ mean_decision_aligned: mean=1.43e-06, max=5.01e-06
      ⚠ queen_decision_aligned: mean=1.67e-06, max=4.12e-06
      ⚠ mean_class_specific: mean=1.43e-06, max=5.01e-06
      ⚠ queen_class_specific: mean=1.67e-06, max=4.12e-06
    
    sharp_boundary: 4/4 configurations passed
      ✓ mean_decision_aligned: mean=2.27e-07, max=6.00e-07
      ✓ queen_decision_aligned: mean=3.79e-07, max=8.01e-07
      ✓ mean_class_specific: mean=2.27e-07, max=6.00e-07
      ✓ queen_class_specific: mean=3.79e-07, max=8.01e-07
    
    clustered_hotspots: 4/4 configurations passed
      ✓ mean_decision_aligned: mean=1.58e-07, max=3.80e-07
      ✓ queen_decision_aligned: mean=2.83e-07, max=4.50e-07
      ✓ mean_class_specific: mean=1.58e-07, max=3.80e-07
      ✓ queen_class_specific: mean=2.83e-07, max=4.50e-07
    
    random_pattern: 4/4 configurations passed
      ✓ mean_decision_aligned: mean=2.47e-07, max=5.30e-07
      ✓ queen_decision_aligned: mean=2.23e-07, max=5.36e-07
      ✓ mean_class_specific: mean=2.47e-07, max=5.30e-07
      ✓ queen_class_specific: mean=2.23e-07, max=5.36e-07
    

Testing GeoSHAP additivity on all group 2 datasets

Note: For convenience, I put in a test status "Passed". The break point for passed is arbitrary. You might want to use a stricter or less strict break point. Also, if you keep my break point, you might want to check if the additivity is reasonable even if it does not "Pass".


```python
geoshap_all_results = {}

for dataset_name in dataset_names:
    print(f"\n{'='*70}")
    print(f"Testing {dataset_name}...")
    print(f"{'='*70}")
    
    results = test_explainer_additivity(
        trainer=trainer_group2,
        analysis_data=analysis_data_group2,
        dataset_name=dataset_name,
        explainer_type='geoshap'
    )
    
    geoshap_all_results[dataset_name] = results

# Summary for all datasets
print("\n" + "=" * 70)
print("GEOSHAP ADDITIVITY SUMMARY - ALL DATASETS")
print("=" * 70)

for dataset_name, results in geoshap_all_results.items():
    passed = sum(1 for r in results.values() if r.get('test_status') == 'PASSED')
    total = len(results)
    print(f"\n{dataset_name}: {passed}/{total} configurations passed")
    
    for config_name, result in results.items():
        if result.get('test_status') == 'PASSED':
            status = "✓" if result['additivity_passed'] else "⚠"
            print(f"  {status} {config_name}: mean={result['mean_additivity_error']:.2e}, "
                  f"max={result['max_additivity_error']:.2e}")

```

    
    ======================================================================
    Testing smooth_gradient...
    ======================================================================
    === ADDITIVITY TEST FOR GEOSHAP - SMOOTH_GRADIENT ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 2
      Baseline: -4.6048
      Additivity error: 3.48e-01
    Testing additivity across 10 samples...
      Mean additivity error: 1.16e-01
      Max additivity error: 3.48e-01
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 2
      Baseline: 25.8321
      Additivity error: 4.32e-04
    Testing additivity across 10 samples...
      Mean additivity error: 7.77e-03
      Max additivity error: 4.62e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=-6.5495, Additivity Error=3.67e-01
      Class 1: Baseline=3.9212, Additivity Error=1.17e+00
      Class 2: Baseline=-4.6048, Additivity Error=3.48e-01
    Testing additivity across 10 samples...
      Mean additivity error: 1.16e-01
      Max additivity error: 3.48e-01
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=-32.4103, Additivity Error=1.19e-03
      Class 1: Baseline=-3.8854, Additivity Error=6.12e-04
      Class 2: Baseline=25.8321, Additivity Error=4.32e-04
    Testing additivity across 10 samples...
      Mean additivity error: 7.77e-03
      Max additivity error: 4.62e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: GEOSHAP - SMOOTH_GRADIENT
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=1.16e-01, Max=3.48e-01
      queen_decision_aligned: ⚠ WARNING - Mean=7.77e-03, Max=4.62e-02
      mean_class_specific: ⚠ WARNING - Mean=1.16e-01, Max=3.48e-01
      queen_class_specific: ⚠ WARNING - Mean=7.77e-03, Max=4.62e-02
    
    ======================================================================
    Testing sharp_boundary...
    ======================================================================
    === ADDITIVITY TEST FOR GEOSHAP - SHARP_BOUNDARY ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: sharp_boundary
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 2
      Baseline: 1.1419
      Additivity error: 3.83e-02
    Testing additivity across 10 samples...
      Mean additivity error: 2.69e-02
      Max additivity error: 3.83e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: sharp_boundary
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 2
      Baseline: 3.4806
      Additivity error: 3.13e-07
    Testing additivity across 10 samples...
      Mean additivity error: 5.78e-04
      Max additivity error: 2.79e-03
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: sharp_boundary
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=0.9024, Additivity Error=2.66e-02
      Class 1: Baseline=3.0175, Additivity Error=9.44e-03
      Class 2: Baseline=1.1419, Additivity Error=3.83e-02
    Testing additivity across 10 samples...
      Mean additivity error: 2.69e-02
      Max additivity error: 3.83e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: sharp_boundary
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=-0.3980, Additivity Error=7.85e-09
      Class 1: Baseline=2.9298, Additivity Error=9.73e-08
      Class 2: Baseline=3.4806, Additivity Error=3.13e-07
    Testing additivity across 10 samples...
      Mean additivity error: 5.78e-04
      Max additivity error: 2.79e-03
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: GEOSHAP - SHARP_BOUNDARY
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=2.69e-02, Max=3.83e-02
      queen_decision_aligned: ⚠ WARNING - Mean=5.78e-04, Max=2.79e-03
      mean_class_specific: ⚠ WARNING - Mean=2.69e-02, Max=3.83e-02
      queen_class_specific: ⚠ WARNING - Mean=5.78e-04, Max=2.79e-03
    
    ======================================================================
    Testing clustered_hotspots...
    ======================================================================
    === ADDITIVITY TEST FOR GEOSHAP - CLUSTERED_HOTSPOTS ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: clustered_hotspots
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 0
      Baseline: -0.5952
      Additivity error: 4.96e-02
    Testing additivity across 10 samples...
      Mean additivity error: 7.37e-02
      Max additivity error: 1.46e-01
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: clustered_hotspots
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 0
      Baseline: -1.4762
      Additivity error: 1.17e-02
    Testing additivity across 10 samples...
      Mean additivity error: 2.84e-03
      Max additivity error: 1.17e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: clustered_hotspots
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=-0.5952, Additivity Error=4.96e-02
      Class 1: Baseline=1.2438, Additivity Error=3.06e-02
      Class 2: Baseline=1.4651, Additivity Error=5.98e-02
    Testing additivity across 10 samples...
      Mean additivity error: 7.37e-02
      Max additivity error: 1.46e-01
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: clustered_hotspots
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=-1.4762, Additivity Error=1.17e-02
      Class 1: Baseline=2.0045, Additivity Error=7.97e-03
      Class 2: Baseline=3.1238, Additivity Error=8.72e-03
    Testing additivity across 10 samples...
      Mean additivity error: 2.84e-03
      Max additivity error: 1.17e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: GEOSHAP - CLUSTERED_HOTSPOTS
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=7.37e-02, Max=1.46e-01
      queen_decision_aligned: ⚠ WARNING - Mean=2.84e-03, Max=1.17e-02
      mean_class_specific: ⚠ WARNING - Mean=7.37e-02, Max=1.46e-01
      queen_class_specific: ⚠ WARNING - Mean=2.84e-03, Max=1.17e-02
    
    ======================================================================
    Testing random_pattern...
    ======================================================================
    === ADDITIVITY TEST FOR GEOSHAP - RANDOM_PATTERN ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: random_pattern
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 0
      Baseline: 0.9583
      Additivity error: 9.67e-02
    Testing additivity across 10 samples...
      Mean additivity error: 5.37e-02
      Max additivity error: 1.92e-01
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: random_pattern
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Auto-detected class: 0
      Baseline: 0.7148
      Additivity error: 1.79e-02
    Testing additivity across 10 samples...
      Mean additivity error: 7.11e-03
      Max additivity error: 1.91e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: random_pattern
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=0.9583, Additivity Error=9.67e-02
      Class 1: Baseline=-0.6415, Additivity Error=1.25e-01
      Class 2: Baseline=-1.1390, Additivity Error=2.48e-01
    Testing additivity across 10 samples...
      Mean additivity error: 5.37e-02
      Max additivity error: 1.92e-01
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
    
    ======================================================================
    GeoSHAP Explainer Initialized: random_pattern
    ======================================================================
    Data Configuration:
      Samples: 10,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: queen
      Explanation mode: class_specific
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
      Class 0: Baseline=0.7148, Additivity Error=1.79e-02
      Class 1: Baseline=-0.5670, Additivity Error=1.38e-02
      Class 2: Baseline=-0.7054, Additivity Error=1.37e-02
    Testing additivity across 10 samples...
      Mean additivity error: 7.11e-03
      Max additivity error: 1.91e-02
      ⚠ Warning: Some additivity errors exceed 1e-6
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY: GEOSHAP - RANDOM_PATTERN
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: ⚠ WARNING - Mean=5.37e-02, Max=1.92e-01
      queen_decision_aligned: ⚠ WARNING - Mean=7.11e-03, Max=1.91e-02
      mean_class_specific: ⚠ WARNING - Mean=5.37e-02, Max=1.92e-01
      queen_class_specific: ⚠ WARNING - Mean=7.11e-03, Max=1.91e-02
    
    ======================================================================
    GEOSHAP ADDITIVITY SUMMARY - ALL DATASETS
    ======================================================================
    
    smooth_gradient: 4/4 configurations passed
      ⚠ mean_decision_aligned: mean=1.16e-01, max=3.48e-01
      ⚠ queen_decision_aligned: mean=7.77e-03, max=4.62e-02
      ⚠ mean_class_specific: mean=1.16e-01, max=3.48e-01
      ⚠ queen_class_specific: mean=7.77e-03, max=4.62e-02
    
    sharp_boundary: 4/4 configurations passed
      ⚠ mean_decision_aligned: mean=2.69e-02, max=3.83e-02
      ⚠ queen_decision_aligned: mean=5.78e-04, max=2.79e-03
      ⚠ mean_class_specific: mean=2.69e-02, max=3.83e-02
      ⚠ queen_class_specific: mean=5.78e-04, max=2.79e-03
    
    clustered_hotspots: 4/4 configurations passed
      ⚠ mean_decision_aligned: mean=7.37e-02, max=1.46e-01
      ⚠ queen_decision_aligned: mean=2.84e-03, max=1.17e-02
      ⚠ mean_class_specific: mean=7.37e-02, max=1.46e-01
      ⚠ queen_class_specific: mean=2.84e-03, max=1.17e-02
    
    random_pattern: 4/4 configurations passed
      ⚠ mean_decision_aligned: mean=5.37e-02, max=1.92e-01
      ⚠ queen_decision_aligned: mean=7.11e-03, max=1.91e-02
      ⚠ mean_class_specific: mean=5.37e-02, max=1.92e-01
      ⚠ queen_class_specific: mean=7.11e-03, max=1.91e-02
    


```python
"""
Diagnostic: Debug GeoSHAP Additivity Errors

This will help us find where the bug is
"""

import numpy as np

# Create a simple GeoSHAP explainer
test_data_group2 = trainer_group2.get_test_data('smooth_gradient')
X_test_group2 = test_data_group2['X_test']

geoshap_test = GeoSHAPExplainer(
    trainer=trainer_group2,
    dataset_name='smooth_gradient',
    X_data=X_test_group2,
    spatial_feature_indices=[8, 9],
    background_type='mean',
    mode='decision_aligned',
    roi_labels=None,
    coords=None,
    grid_size=None
)

# Test one sample
sample_idx = 0
result = geoshap_test.explain_sample(sample_idx)

print("GEOSHAP ADDITIVITY DIAGNOSTIC")
print("=" * 60)
print(f"\nSample {sample_idx}:")
print(f"  Baseline (f_background): {result['baseline']:.6f}")
print(f"  φ_GEO: {result['phi_GEO']:.6f}")
print(f"  Σφ_features: {np.sum(result['phi_features']):.6f}")
print(f"  Σφ_interactions: {np.sum(result['phi_interactions']):.6f}")
print(f"\n  Sum of components: {result['baseline'] + result['phi_GEO'] + np.sum(result['phi_features']) + np.sum(result['phi_interactions']):.6f}")
print(f"  Actual f(x): {result['full_value']:.6f}")
print(f"  Predicted f(x): {result['predicted_full']:.6f}")
print(f"  Error: {result['additivity_error']:.6f}")

# Check if the issue is in the baseline computation
print(f"\n\nDETAILED BREAKDOWN:")
print(f"  1. Baseline value: {result['baseline']:.6f}")

# Manually verify baseline
sample = X_test_group2[sample_idx]
background = geoshap_test._get_background_for_sample(sample_idx)
baseline_check = geoshap_test._get_model_output(background.reshape(1, -1), output_type='logits')
print(f"  2. Manual baseline check: {baseline_check[0, result['target_class']]:.6f}")

# Check full value
full_check = geoshap_test._get_model_output(sample.reshape(1, -1), output_type='logits')
print(f"  3. Manual full value check: {full_check[0, result['target_class']]:.6f}")

# The difference
print(f"\n  Full - Baseline = {full_check[0, result['target_class']] - baseline_check[0, result['target_class']]:.6f}")
print(f"  φ_GEO + Σφ_j + Σφ_(GEO,j) = {result['phi_GEO'] + np.sum(result['phi_features']) + np.sum(result['phi_interactions']):.6f}")
print(f"  Difference = {abs((full_check[0, result['target_class']] - baseline_check[0, result['target_class']]) - (result['phi_GEO'] + np.sum(result['phi_features']) + np.sum(result['phi_interactions']))):.6f}")

```

    
    ======================================================================
    GeoSHAP Explainer Initialized: smooth_gradient
    ======================================================================
    Data Configuration:
      Samples: 2,000
      Total features: 10
      Spatial features (joint): [8, 9] (n=2)
      Non-spatial features: [0, 1, 2, 3, 4, 5, 6, 7] (n=8)
      Classes: 3
    
    GeoSHAP Configuration:
      Background type: mean
      Explanation mode: decision_aligned
      Coalition space: 2^9 = 512 coalitions
      (vs. standard SHAP: 2^10 = 1,024 coalitions)
    ======================================================================
    
    Pre-computing GeoSHAP coalition structures...
      φ_GEO: 256 coalitions
      φ_j: 128 coalitions per feature
      φ_(GEO,j): 128 coalitions per interaction
    Total coalition structures cached: 17
    GEOSHAP ADDITIVITY DIAGNOSTIC
    ============================================================
    
    Sample 0:
      Baseline (f_background): -4.794620
      φ_GEO: 0.791789
      Σφ_features: 13.368400
      Σφ_interactions: -0.050337
    
      Sum of components: 9.315233
      Actual f(x): 9.273224
      Predicted f(x): 9.315233
      Error: 0.042009
    
    
    DETAILED BREAKDOWN:
      1. Baseline value: -4.794620
      2. Manual baseline check: -4.794620
      3. Manual full value check: 9.273224
    
      Full - Baseline = 14.067844
      φ_GEO + Σφ_j + Σφ_(GEO,j) = 14.109853
      Difference = 0.042008
    


```python
# Test GeoSHAP additivity on Group 2 data
geoshap_test_results = test_explainer_additivity(
    trainer=trainer_group2,
    analysis_data=analysis_data_group2,
    dataset_name='smooth_gradient',
    explainer_type='geoshap'
)
```


```python
def comprehensive_shap_test(trainer, analysis_data, dataset_name='smooth_gradient'):
    """
    Comprehensive test of all 4 SHAP configurations for a dataset
    
    Parameters:
    - trainer: Your SimpleMLPModels instance
    - analysis_data: Results from setup_consistent_samples_for_analysis
    - dataset_name: Which dataset to test
    
    Returns:
    - test_results: Dictionary with results from all configurations
    """
    print(f"=== COMPREHENSIVE SHAP TEST FOR {dataset_name.upper()} ===")
    print("=" * 60)
    
    # Get the consistent samples for this dataset
    dataset_info = analysis_data[dataset_name]
    selected_samples = dataset_info['selected_samples']
    dataset_dict = dataset_info['dataset']
    
    # Test configurations: [background_type, mode]
    configurations = [
        ('mean', 'decision_aligned'),      # Global + Decision-aligned
        ('queen', 'decision_aligned'),     # Queen + Decision-aligned  
        ('mean', 'class_specific'),        # Global + Class-specific
        ('queen', 'class_specific')        # Queen + Class-specific
    ]
    
    test_results = {}
    
    for config_idx, (bg_type, mode) in enumerate(configurations, 1):
        config_name = f"{bg_type}_{mode}"
        print(f"\n--- Configuration {config_idx}: {bg_type.title()} Background + {mode.replace('_', ' ').title()} ---")
        
        try:
            # Initialize explainer for this configuration
            explainer = CustomSHAPExplainer(
                trainer=trainer,
                dataset_name=dataset_name,
                X_data=dataset_dict['X'],
                roi_labels=dataset_dict['roi_labels'], 
                coords=dataset_dict['coords'],
                grid_size=dataset_dict['grid_size'],
                background_type=bg_type,
                mode=mode
            )
            
            # Test basic functionality
            print(f"Testing basic functionality...")
            
            # Get first selected sample for testing
            first_sample_idx = selected_samples['sample_indices'][0]
            
            # For class-specific mode, test with each class
            if mode == 'class_specific':
                print(f"Testing class-specific mode with all classes...")
                class_results = []
                
                for target_class in range(explainer.n_classes):
                    result = explainer.explain_sample(first_sample_idx, target_class=target_class)
                    class_results.append(result)
                    print(f"  Class {target_class}: Baseline={result['baseline']:.4f}, "
                          f"Additivity Error={result['additivity_error']:.2e}")
                
                # Use class 0 for subsequent tests
                test_result = class_results[0]
            else:
                # Decision-aligned mode
                test_result = explainer.explain_sample(first_sample_idx)
                print(f"  Auto-detected class: {test_result['target_class']}")
                print(f"  Baseline: {test_result['baseline']:.4f}")
                print(f"  Additivity error: {test_result['additivity_error']:.2e}")
            
            # Test additivity across multiple samples (for information only)
            print(f"Testing additivity across 10 samples...")
            additivity_errors = []
            
            for i in range(min(10, len(selected_samples['sample_indices']))):
                sample_idx = selected_samples['sample_indices'][i]
                
                if mode == 'class_specific':
                    # Use predicted class for consistency
                    sample_pred = trainer.get_probabilities_prediction(
                        dataset_name, 
                        dataset_dict['X'][sample_idx:sample_idx+1]
                    )
                    pred_class = np.argmax(sample_pred[0])
                    result = explainer.explain_sample(sample_idx, target_class=pred_class)
                else:
                    result = explainer.explain_sample(sample_idx)
                
                additivity_errors.append(result['additivity_error'])
            
            print(f"  Mean additivity error: {np.mean(additivity_errors):.2e}")
            print(f"  Max additivity error: {np.max(additivity_errors):.2e}")
            
            # Test ROI explanation (small subset for speed)
            print(f"Testing ROI explanation...")
            roi_id = selected_samples['roi_indices'][0]
            
            if mode == 'class_specific':
                # Use most common predicted class
                roi_samples = selected_samples['X_selected'][:5]  # First 5 for speed
                roi_preds = trainer.get_probabilities_prediction(dataset_name, roi_samples)
                common_class = np.bincount(np.argmax(roi_preds, axis=1)).argmax()
                roi_result = explainer.explain_roi(roi_id, n_samples=3, target_class=common_class)
            else:
                roi_result = explainer.explain_roi(roi_id, n_samples=3)
            
            print(f"  ROI {roi_id}: {roi_result['n_samples']} samples explained")
            print(f"  Mean SHAP values: {np.round(roi_result['mean_shap_values'], 4)}")
            print(f"  Mean additivity error: {roi_result['mean_additivity_error']:.2e}")
            
            # Store results
            test_results[config_name] = {
                'configuration': (bg_type, mode),
                'explainer': explainer,
                'basic_test_result': test_result,
                'additivity_errors': additivity_errors,
                'roi_test_result': roi_result,
                'mean_additivity_error': np.mean(additivity_errors),
                'max_additivity_error': np.max(additivity_errors),
                'additivity_passed': np.all(np.array(additivity_errors) < 1e-6),
                'test_status': 'PASSED'
            }
            
            print(f"✓ Configuration {config_name} PASSED")
            
        except Exception as e:
            print(f"✗ Configuration {config_name} FAILED: {str(e)}")
            test_results[config_name] = {
                'configuration': (bg_type, mode),
                'test_status': 'FAILED',
                'error': str(e)
            }
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"TEST SUMMARY FOR {dataset_name.upper()}")
    print(f"=" * 60)
    
    passed_configs = [name for name, result in test_results.items() 
                     if result.get('test_status') == 'PASSED']
    failed_configs = [name for name, result in test_results.items() 
                     if result.get('test_status') == 'FAILED']
    
    print(f"Configurations tested: {len(test_results)}")
    print(f"Passed: {len(passed_configs)} - {passed_configs}")
    print(f"Failed: {len(failed_configs)} - {failed_configs}")
    
    if passed_configs:
        print(f"\nAdditivity Check Summary:")
        for config_name in passed_configs:
            result = test_results[config_name]
            print(f"  {config_name}: Mean error = {result['mean_additivity_error']:.2e}, "
                  f"Max error = {result['max_additivity_error']:.2e}")
    
    return test_results


def test_configuration_differences(test_results):
    """
    Analyze differences between SHAP configurations
    
    Parameters:
    - test_results: Results from comprehensive_shap_test
    """
    print(f"\n=== CONFIGURATION COMPARISON ANALYSIS ===")
    print("=" * 50)
    
    passed_results = {name: result for name, result in test_results.items() 
                     if result.get('test_status') == 'PASSED'}
    
    if len(passed_results) < 2:
        print("Need at least 2 passing configurations for comparison")
        return
    
    # Compare SHAP values from basic tests
    print("Comparing SHAP values from basic tests:")
    
    for name1, result1 in passed_results.items():
        for name2, result2 in passed_results.items():
            if name1 < name2:  # Avoid duplicate comparisons
                shap1 = result1['basic_test_result']['shap_values']
                shap2 = result2['basic_test_result']['shap_values']
                
                # Calculate similarity metrics
                correlation = np.corrcoef(shap1, shap2)[0, 1]
                rmse = np.sqrt(np.mean((shap1 - shap2) ** 2))
                max_diff = np.max(np.abs(shap1 - shap2))
                
                print(f"\n{name1} vs {name2}:")
                print(f"  Correlation: {correlation:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  Max difference: {max_diff:.4f}")
                print(f"  SHAP1: {np.round(shap1, 4)}")
                print(f"  SHAP2: {np.round(shap2, 4)}")
    
    return passed_results

def run_full_shap_analysis_on_selected_samples(explainer, selected_samples, target_class=None):
    """
    Run SHAP analysis on all selected samples (one per ROI)
    
    Parameters:
    - explainer: CustomSHAPExplainer instance
    - selected_samples: Selected samples from setup_consistent_samples_for_analysis
    - target_class: For class-specific mode (None for decision-aligned)
    
    Returns:
    - shap_results: Array of SHAP values for all selected samples
    """
    print(f"\nRunning SHAP analysis on {selected_samples['n_samples']} selected samples...")
    print(f"Configuration: {explainer.background_type} background, {explainer.mode} mode")
    print(f"Expected time: ~{selected_samples['n_samples'] * 3:.0f} seconds")
    
    all_shap_values = []
    all_results = []
    
    from tqdm import tqdm
    
    for i, sample_idx in enumerate(tqdm(selected_samples['sample_indices'], desc="Computing SHAP")):
        if explainer.mode == 'class_specific' and target_class is None:
            # Cycle through classes for class-specific mode
            cycle_target_class = i % explainer.n_classes
            result = explainer.explain_sample(sample_idx, target_class=cycle_target_class)
        else:
            result = explainer.explain_sample(sample_idx, target_class=target_class)
        
        all_shap_values.append(result['shap_values'])
        all_results.append(result)
    
    shap_array = np.array(all_shap_values)
    
    print(f"Completed SHAP analysis:")
    print(f"  Shape: {shap_array.shape}")
    print(f"  Mean additivity error: {np.mean([r['additivity_error'] for r in all_results]):.2e}")
    print(f"  Max additivity error: {np.max([r['additivity_error'] for r in all_results]):.2e}")

    return {
        'shap_values': shap_array,
        'individual_results': all_results,
        'selected_samples': selected_samples,
        'configuration': (explainer.background_type, explainer.mode)
    }
```

Step 1: Comprehensive GeoSHAP Test Function

This function tests all 4 GeoSHAP configurations for a Group 2 dataset:
1. Mean Background + Decision-Aligned
2. Queen Background + Decision-Aligned
3. Mean Background + Class-Specific
4. Queen Background + Class-Specific

It verifies:
- Basic functionality (single sample explanation)
- Additivity property (should be < 1e-6)
- ROI-level explanation
- All configurations work correctly


```python
def comprehensive_geoshap_test(trainer, analysis_data, dataset_name='smooth_gradient_group2'):
    """
    Comprehensive test of all 4 GeoSHAP configurations for a Group 2 dataset
    
    Parameters:
    -----------
    trainer : SimpleMLPModels instance
        Your trained model wrapper
    analysis_data : dict
        Results from setup_consistent_samples_for_analysis
        Should contain analysis_data[dataset_name] with:
            - 'selected_samples': Selected sample info
            - 'dataset': Dataset dictionary with X, roi_labels, coords, grid_size
    dataset_name : str
        Which Group 2 dataset to test (e.g., 'smooth_gradient_group2')
    
    Returns:
    --------
    test_results : dict
        Dictionary with results from all 4 configurations
        Keys: 'mean_decision_aligned', 'queen_decision_aligned', 
              'mean_class_specific', 'queen_class_specific'
    """
    print(f"=== COMPREHENSIVE GEOSHAP TEST FOR {dataset_name.upper()} ===")
    print("=" * 60)
    
    # Get the consistent samples for this dataset
    dataset_info = analysis_data[dataset_name]
    selected_samples = dataset_info['selected_samples']
    dataset_dict = dataset_info['dataset']
    
    # Verify this is Group 2 data (should have 10 features)
    if dataset_dict['X'].shape[1] != 10:
        raise ValueError(f"GeoSHAP requires Group 2 data (10 features), "
                        f"but got {dataset_dict['X'].shape[1]} features")
    
    # Test configurations: [background_type, mode]
    configurations = [
        ('mean', 'decision_aligned'),      # Global + Decision-aligned
        ('queen', 'decision_aligned'),     # Queen + Decision-aligned  
        ('mean', 'class_specific'),        # Global + Class-specific
        ('queen', 'class_specific')        # Queen + Class-specific
    ]
    
    test_results = {}
    
    for config_idx, (bg_type, mode) in enumerate(configurations, 1):
        config_name = f"{bg_type}_{mode}"
        print(f"\n--- Configuration {config_idx}: {bg_type.title()} Background + "
              f"{mode.replace('_', ' ').title()} ---")
        
        try:
            # Initialize GeoSHAP explainer for this configuration
            from geoshap_explainer_v1 import GeoSHAPExplainer
            
            explainer = GeoSHAPExplainer(
                trainer=trainer,
                dataset_name=dataset_name,
                X_data=dataset_dict['X'],
                spatial_feature_indices=[8, 9],  # x, y coordinates
                background_type=bg_type,
                mode=mode,
                roi_labels=dataset_dict['roi_labels'], 
                coords=dataset_dict['coords'],
                grid_size=dataset_dict['grid_size']
            )
            
            # Test basic functionality
            print(f"Testing basic functionality...")
            
            # Get first selected sample for testing
            first_sample_idx = selected_samples['sample_indices'][0]
            
            # For class-specific mode, test with each class
            if mode == 'class_specific':
                print(f"Testing class-specific mode with all classes...")
                class_results = []
                
                for target_class in range(explainer.n_classes):
                    result = explainer.explain_sample(first_sample_idx, target_class=target_class)
                    class_results.append(result)
                    print(f"  Class {target_class}: Baseline={result['baseline']:.4f}, "
                          f"φ_GEO={result['phi_GEO']:.4f}, "
                          f"Additivity Error={result['additivity_error']:.2e}")
                
                # Use class 0 for subsequent tests
                test_result = class_results[0]
            else:
                # Decision-aligned mode
                test_result = explainer.explain_sample(first_sample_idx)
                print(f"  Auto-detected class: {test_result['target_class']}")
                print(f"  Baseline: {test_result['baseline']:.4f}")
                print(f"  φ_GEO (location effect): {test_result['phi_GEO']:.4f}")
                print(f"  Σφ_features: {np.sum(test_result['phi_features']):.4f}")
                print(f"  Σφ_interactions: {np.sum(test_result['phi_interactions']):.4f}")
                print(f"  Additivity error: {test_result['additivity_error']:.2e}")
            
            # Test additivity across multiple samples (for information only)
            print(f"Testing additivity across 10 samples...")
            additivity_errors = []
            
            for i in range(min(10, len(selected_samples['sample_indices']))):
                sample_idx = selected_samples['sample_indices'][i]
                
                if mode == 'class_specific':
                    # Use predicted class for consistency
                    sample_pred = trainer.get_probabilities_prediction(
                        dataset_name, 
                        dataset_dict['X'][sample_idx:sample_idx+1]
                    )
                    pred_class = np.argmax(sample_pred[0])
                    result = explainer.explain_sample(sample_idx, target_class=pred_class)
                else:
                    result = explainer.explain_sample(sample_idx)
                
                additivity_errors.append(result['additivity_error'])
            
            print(f"  Mean additivity error: {np.mean(additivity_errors):.2e}")
            print(f"  Max additivity error: {np.max(additivity_errors):.2e}")
            
            # Verify additivity is good (< 1e-6)
            additivity_passed = np.all(np.array(additivity_errors) < 1e-6)
            if additivity_passed:
                print(f"  ✓ Additivity check PASSED")
            else:
                print(f"  ⚠ Warning: Some additivity errors exceed 1e-6")
            
            # Store results
            test_results[config_name] = {
                'configuration': (bg_type, mode),
                'explainer': explainer,
                'basic_test_result': test_result,
                'additivity_errors': additivity_errors,
                'mean_additivity_error': np.mean(additivity_errors),
                'max_additivity_error': np.max(additivity_errors),
                'additivity_passed': additivity_passed,
                'test_status': 'PASSED'
            }
            
            print(f"✓ Configuration {config_name} PASSED")
            
        except Exception as e:
            print(f"✗ Configuration {config_name} FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            
            test_results[config_name] = {
                'configuration': (bg_type, mode),
                'test_status': 'FAILED',
                'error': str(e)
            }
    
    # Summary
    print(f"\n" + "=" * 60)
    print(f"TEST SUMMARY FOR {dataset_name.upper()}")
    print(f"=" * 60)
    
    passed_configs = [name for name, result in test_results.items() 
                     if result.get('test_status') == 'PASSED']
    failed_configs = [name for name, result in test_results.items() 
                     if result.get('test_status') == 'FAILED']
    
    print(f"Configurations tested: {len(test_results)}")
    print(f"Passed: {len(passed_configs)} - {passed_configs}")
    print(f"Failed: {len(failed_configs)} - {failed_configs}")
    
    if passed_configs:
        print(f"\nAdditivity Check Summary:")
        for config_name in passed_configs:
            result = test_results[config_name]
            print(f"  {config_name}: Mean error = {result['mean_additivity_error']:.2e}, "
                  f"Max error = {result['max_additivity_error']:.2e}")
    
    return test_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
"""
# After you've run setup_consistent_samples_for_analysis for Group 2 datasets:

# Test GeoSHAP on smooth_gradient_group2
geoshap_test_results = comprehensive_geoshap_test(
    trainer=trainer,
    analysis_data=analysis_data,
    dataset_name='smooth_gradient_group2'
)

# Check if all configurations passed
for config_name, result in geoshap_test_results.items():
    if result['test_status'] == 'PASSED':
        print(f"{config_name}: ✓ PASSED")
    else:
        print(f"{config_name}: ✗ FAILED - {result['error']}")
"""
```

## SHAP Test for Group 1


```python
print("\n" + "="*70)
print("TESTING SHAP CONFIGURATIONS - GROUP 1")
print("="*70)

# Test on one dataset first (choosing smooth_gradient since it performs well)
shap_test_results_group1 = comprehensive_shap_test(
    trainer=trainer_group1,
    analysis_data=analysis_data_group1,
    dataset_name='smooth_gradient'
)

# Analyze differences between configurations
config_comparison = test_configuration_differences(shap_test_results_group1)
```

    
    ======================================================================
    TESTING SHAP CONFIGURATIONS - GROUP 1
    ======================================================================
    === COMPREHENSIVE SHAP TEST FOR SMOOTH_GRADIENT ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: -5.9359
      Additivity error: 1.43e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.39e-06
      Max additivity error: 3.20e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.87s/it]
    

      ROI 0: 3 samples explained
      Mean SHAP values: [ 7.541   6.0539  3.5362 10.3541  6.0978  2.5747 -1.4882 -1.5684]
      Mean additivity error: 3.18e-06
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: decision_aligned
      Features: 8, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Auto-detected class: 2
      Baseline: 24.2444
      Additivity error: 2.57e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.11e-06
      Max additivity error: 5.58e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.99s/it]
    

      ROI 0: 3 samples explained
      Mean SHAP values: [ 0.9845  0.8994  0.6549  1.0375  0.4544  0.3858 -0.2365 -0.127 ]
      Mean additivity error: 8.84e-07
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-9.9147, Additivity Error=1.14e-05
      Class 1: Baseline=4.5595, Additivity Error=2.98e-07
      Class 2: Baseline=-5.9359, Additivity Error=1.43e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.39e-06
      Max additivity error: 3.20e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.82s/it]
    

      ROI 0: 3 samples explained
      Mean SHAP values: [ 7.8601  6.2354  4.0025 11.2258  6.0971  2.5623 -1.5054 -1.4071]
      Mean additivity error: 2.30e-06
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: class_specific
      Features: 8, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 1024 coalition patterns
      Class 0: Baseline=-55.8512, Additivity Error=9.51e-06
      Class 1: Baseline=-2.7355, Additivity Error=7.02e-07
      Class 2: Baseline=24.2444, Additivity Error=2.57e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.11e-06
      Max additivity error: 5.58e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.69s/it]

      ROI 0: 3 samples explained
      Mean SHAP values: [ 0.9761  1.204   0.4491  1.7162  0.7487  0.4639 -0.1445 -0.1303]
      Mean additivity error: 1.66e-06
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY FOR SMOOTH_GRADIENT
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: Mean error = 1.39e-06, Max error = 3.20e-06
      queen_decision_aligned: Mean error = 1.11e-06, Max error = 5.58e-06
      mean_class_specific: Mean error = 1.39e-06, Max error = 3.20e-06
      queen_class_specific: Mean error = 1.11e-06, Max error = 5.58e-06
    
    === CONFIGURATION COMPARISON ANALYSIS ===
    ==================================================
    Comparing SHAP values from basic tests:
    
    mean_decision_aligned vs queen_decision_aligned:
      Correlation: 0.9275
      RMSE: 5.2313
      Max difference: 9.7037
      SHAP1: [ 8.2747  5.4551  4.1402 11.5402  6.3525  2.452  -1.5512 -1.4332]
      SHAP2: [ 1.331   0.1754  0.8502  1.8365  0.9511  0.2023 -0.1763 -0.1201]
    
    mean_decision_aligned vs queen_class_specific:
      Correlation: -0.9338
      RMSE: 7.8793
      Max difference: 14.7846
      SHAP1: [ 8.2747  5.4551  4.1402 11.5402  6.3525  2.452  -1.5512 -1.4332]
      SHAP2: [-2.9135 -0.3667 -1.7947 -3.2443 -1.7361 -0.3471  0.9927  0.5827]
    
    mean_class_specific vs mean_decision_aligned:
      Correlation: -0.9840
      RMSE: 16.7632
      Max difference: 29.6460
      SHAP1: [-16.447  -10.659   -7.2171 -18.1058  -9.4172  -3.2073   5.5992   4.6909]
      SHAP2: [ 8.2747  5.4551  4.1402 11.5402  6.3525  2.452  -1.5512 -1.4332]
    
    mean_class_specific vs queen_decision_aligned:
      Correlation: -0.8876
      RMSE: 11.5750
      Max difference: 19.9423
      SHAP1: [-16.447  -10.659   -7.2171 -18.1058  -9.4172  -3.2073   5.5992   4.6909]
      SHAP2: [ 1.331   0.1754  0.8502  1.8365  0.9511  0.2023 -0.1763 -0.1201]
    
    mean_class_specific vs queen_class_specific:
      Correlation: 0.9236
      RMSE: 8.9766
      Max difference: 14.8615
      SHAP1: [-16.447  -10.659   -7.2171 -18.1058  -9.4172  -3.2073   5.5992   4.6909]
      SHAP2: [-2.9135 -0.3667 -1.7947 -3.2443 -1.7361 -0.3471  0.9927  0.5827]
    
    queen_class_specific vs queen_decision_aligned:
      Correlation: -0.9834
      RMSE: 2.7501
      Max difference: 5.0809
      SHAP1: [-2.9135 -0.3667 -1.7947 -3.2443 -1.7361 -0.3471  0.9927  0.5827]
      SHAP2: [ 1.331   0.1754  0.8502  1.8365  0.9511  0.2023 -0.1763 -0.1201]
    

    
    

## Run full SHAP analysis

Run the full SHAP analysis on all selected samples for the mean-decision aligned and queen-decision aligned configurations.


```python
print("\n" + "="*70)
print("RUNNING FULL SHAP ANALYSIS - GROUP 1")
print("="*70)

# Use the explainers from the test results
mean_explainer = shap_test_results_group1['mean_decision_aligned']['explainer']
queen_explainer = shap_test_results_group1['queen_decision_aligned']['explainer']

# Run full SHAP analysis for both configurations
print("\n### Configuration 1: Mean Background + Decision Aligned ###")
shap_results_mean = run_full_shap_analysis_on_selected_samples(
    explainer=mean_explainer,
    selected_samples=analysis_data_group1['smooth_gradient']['selected_samples']
)

print("\n### Configuration 2: Queen Background + Decision Aligned ###")
shap_results_queen = run_full_shap_analysis_on_selected_samples(
    explainer=queen_explainer,
    selected_samples=analysis_data_group1['smooth_gradient']['selected_samples']
)

print("\n" + "="*70)
print("FULL SHAP ANALYSIS COMPLETE")
print("="*70)
```

    
    ======================================================================
    RUNNING FULL SHAP ANALYSIS - GROUP 1
    ======================================================================
    
    ### Configuration 1: Mean Background + Decision Aligned ###
    
    Running SHAP analysis on 100 selected samples...
    Configuration: mean background, decision_aligned mode
    Expected time: ~300 seconds
    

    Computing SHAP: 100%|████████████████████████████████████████████████████████████████| 100/100 [04:58<00:00,  2.98s/it]
    

    Completed SHAP analysis:
      Shape: (100, 8)
      Mean additivity error: 1.39e-06
      Max additivity error: 5.13e-06
    
    ### Configuration 2: Queen Background + Decision Aligned ###
    
    Running SHAP analysis on 100 selected samples...
    Configuration: queen background, decision_aligned mode
    Expected time: ~300 seconds
    

    Computing SHAP: 100%|████████████████████████████████████████████████████████████████| 100/100 [04:44<00:00,  2.85s/it]

    Completed SHAP analysis:
      Shape: (100, 8)
      Mean additivity error: 1.28e-06
      Max additivity error: 7.64e-06
    
    ======================================================================
    FULL SHAP ANALYSIS COMPLETE
    ======================================================================
    

    
    

## SHAP Test for Group 2


```python
print("\n" + "="*70)
print("TESTING SHAP CONFIGURATIONS - GROUP 2")
print("="*70)

# Test on one dataset first (choosing smooth_gradient since it performs well)
shap_test_results_group2 = comprehensive_shap_test(
    trainer=trainer_group2,
    analysis_data=analysis_data_group2,
    dataset_name='smooth_gradient'
)

# Analyze differences between configurations
config_comparison = test_configuration_differences(shap_test_results_group2)
```

    
    ======================================================================
    TESTING SHAP CONFIGURATIONS - GROUP 2
    ======================================================================
    === COMPREHENSIVE SHAP TEST FOR SMOOTH_GRADIENT ===
    ============================================================
    
    --- Configuration 1: Mean Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 2
      Baseline: -4.6048
      Additivity error: 5.01e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.43e-06
      Max additivity error: 5.01e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:14<00:00,  4.75s/it]
    

      ROI 0: 3 samples explained
      Mean SHAP values: [ 7.8652  5.4634  3.2314  9.3624  6.0811  1.4807 -0.3883 -0.3363  1.243
      1.114 ]
      Mean additivity error: 1.16e-06
    ✓ Configuration mean_decision_aligned PASSED
    
    --- Configuration 2: Queen Background + Decision Aligned ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: decision_aligned
      Features: 10, Classes: 3
    Testing basic functionality...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Auto-detected class: 2
      Baseline: 25.8321
      Additivity error: 3.79e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.67e-06
      Max additivity error: 4.12e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:13<00:00,  4.45s/it]
    

      ROI 0: 3 samples explained
      Mean SHAP values: [ 0.8419  0.6559  0.5835  1.3121  0.6251  0.2655 -0.0511 -0.0542  0.1568
      0.1485]
      Mean additivity error: 4.84e-07
    ✓ Configuration queen_decision_aligned PASSED
    
    --- Configuration 3: Mean Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: mean
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=-6.5495, Additivity Error=5.72e-06
      Class 1: Baseline=3.9212, Additivity Error=8.94e-08
      Class 2: Baseline=-4.6048, Additivity Error=5.01e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.43e-06
      Max additivity error: 5.01e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:13<00:00,  4.43s/it]
    

      ROI 0: 3 samples explained
      Mean SHAP values: [ 7.9508  5.4046  3.4751  9.9501  5.5962  1.5101 -0.4027 -0.3158  1.2424
      1.1132]
      Mean additivity error: 3.02e-06
    ✓ Configuration mean_class_specific PASSED
    
    --- Configuration 4: Queen Background + Class Specific ---
      CustomSHAPExplainer initialized for smooth_gradient
      Background type: queen
      Mode: class_specific
      Features: 10, Classes: 3
    Testing basic functionality...
    Testing class-specific mode with all classes...
    Pre-computing coalition structures...
    Cached 5120 coalition patterns
      Class 0: Baseline=-32.4103, Additivity Error=4.19e-06
      Class 1: Baseline=-3.8854, Additivity Error=1.53e-07
      Class 2: Baseline=25.8321, Additivity Error=3.79e-06
    Testing additivity across 10 samples...
      Mean additivity error: 1.67e-06
      Max additivity error: 4.12e-06
    Testing ROI explanation...
    Computing exact SHAP for 3 samples in ROI 0
    

    ROI 0: 100%|█████████████████████████████████████████████████████████████████████████████| 3/3 [00:12<00:00,  4.11s/it]

      ROI 0: 3 samples explained
      Mean SHAP values: [ 0.9953  0.8338  0.2941  1.2335  0.9741  0.085  -0.0701 -0.0534  0.1562
      0.1483]
      Mean additivity error: 1.38e-06
    ✓ Configuration queen_class_specific PASSED
    
    ============================================================
    TEST SUMMARY FOR SMOOTH_GRADIENT
    ============================================================
    Configurations tested: 4
    Passed: 4 - ['mean_decision_aligned', 'queen_decision_aligned', 'mean_class_specific', 'queen_class_specific']
    Failed: 0 - []
    
    Additivity Check Summary:
      mean_decision_aligned: Mean error = 1.43e-06, Max error = 5.01e-06
      queen_decision_aligned: Mean error = 1.67e-06, Max error = 4.12e-06
      mean_class_specific: Mean error = 1.43e-06, Max error = 5.01e-06
      queen_class_specific: Mean error = 1.67e-06, Max error = 4.12e-06
    
    === CONFIGURATION COMPARISON ANALYSIS ===
    ==================================================
    Comparing SHAP values from basic tests:
    
    mean_decision_aligned vs queen_decision_aligned:
      Correlation: 0.9220
      RMSE: 4.1826
      Max difference: 8.2766
      SHAP1: [ 8.2617  5.0823  3.8204  9.8617  5.7429  1.4242 -0.3957 -0.3238  1.2411
      1.1105]
      SHAP2: [ 1.3727  0.1776  0.934   1.5851  0.986   0.1311 -0.0716 -0.0313  0.1564
      0.1483]
    
    mean_decision_aligned vs queen_class_specific:
      Correlation: -0.9040
      RMSE: 5.6964
      Max difference: 11.1922
      SHAP1: [ 8.2617  5.0823  3.8204  9.8617  5.7429  1.4242 -0.3957 -0.3238  1.2411
      1.1105]
      SHAP2: [-1.7879 -0.1949 -0.9004 -1.3305 -0.716  -0.126   0.2938  0.1941 -0.1092
     -0.0749]
    
    mean_class_specific vs mean_decision_aligned:
      Correlation: -0.9513
      RMSE: 9.6349
      Max difference: 18.4589
      SHAP1: [-10.1973  -5.5619  -3.7413  -7.5145  -3.9963  -1.1349   1.4875   1.4429
      -0.822   -0.5749]
      SHAP2: [ 8.2617  5.0823  3.8204  9.8617  5.7429  1.4242 -0.3957 -0.3238  1.2411
      1.1105]
    
    mean_class_specific vs queen_decision_aligned:
      Correlation: -0.8541
      RMSE: 5.5134
      Max difference: 11.5700
      SHAP1: [-10.1973  -5.5619  -3.7413  -7.5145  -3.9963  -1.1349   1.4875   1.4429
      -0.822   -0.5749]
      SHAP2: [ 1.3727  0.1776  0.934   1.5851  0.986   0.1311 -0.0716 -0.0313  0.1564
      0.1483]
    
    mean_class_specific vs queen_class_specific:
      Correlation: 0.9231
      RMSE: 4.0169
      Max difference: 8.4094
      SHAP1: [-10.1973  -5.5619  -3.7413  -7.5145  -3.9963  -1.1349   1.4875   1.4429
      -0.822   -0.5749]
      SHAP2: [-1.7879 -0.1949 -0.9004 -1.3305 -0.716  -0.126   0.2938  0.1941 -0.1092
     -0.0749]
    
    queen_class_specific vs queen_decision_aligned:
      Correlation: -0.9588
      RMSE: 1.5894
      Max difference: 3.1606
      SHAP1: [-1.7879 -0.1949 -0.9004 -1.3305 -0.716  -0.126   0.2938  0.1941 -0.1092
     -0.0749]
      SHAP2: [ 1.3727  0.1776  0.934   1.5851  0.986   0.1311 -0.0716 -0.0313  0.1564
      0.1483]
    

    
    

## Run full SHAP analysis for both groups


```python
print("\n" + "="*70)
print("RUNNING FULL SHAP ANALYSIS - GROUP 1")
print("="*70)

# Group 1: Mean and Queen explainers
mean_explainer_g1 = shap_test_results_group1['mean_decision_aligned']['explainer']
queen_explainer_g1 = shap_test_results_group1['queen_decision_aligned']['explainer']

print("\n### Group 1: Mean Background ###")
shap_results_mean_g1 = run_full_shap_analysis_on_selected_samples(
    explainer=mean_explainer_g1,
    selected_samples=analysis_data_group1['smooth_gradient']['selected_samples']
)

print("\n### Group 1: Queen Background ###")
shap_results_queen_g1 = run_full_shap_analysis_on_selected_samples(
    explainer=queen_explainer_g1,
    selected_samples=analysis_data_group1['smooth_gradient']['selected_samples']
)

print("\n" + "="*70)
print("RUNNING FULL SHAP ANALYSIS - GROUP 2")
print("="*70)

# Group 2: Mean and Queen explainers
mean_explainer_g2 = shap_test_results_group2['mean_decision_aligned']['explainer']
queen_explainer_g2 = shap_test_results_group2['queen_decision_aligned']['explainer']

print("\n### Group 2: Mean Background ###")
shap_results_mean_g2 = run_full_shap_analysis_on_selected_samples(
    explainer=mean_explainer_g2,
    selected_samples=analysis_data_group2['smooth_gradient']['selected_samples']
)

print("\n### Group 2: Queen Background ###")
shap_results_queen_g2 = run_full_shap_analysis_on_selected_samples(
    explainer=queen_explainer_g2,
    selected_samples=analysis_data_group2['smooth_gradient']['selected_samples']
)

print("\n" + "="*70)
print("FULL SHAP ANALYSIS COMPLETE FOR BOTH GROUPS")
print("="*70)
```

    
    ======================================================================
    RUNNING FULL SHAP ANALYSIS - GROUP 1
    ======================================================================
    
    ### Group 1: Mean Background ###
    
    Running SHAP analysis on 100 selected samples...
    Configuration: mean background, decision_aligned mode
    Expected time: ~300 seconds
    

    Computing SHAP: 100%|████████████████████████████████████████████████████████████████| 100/100 [04:37<00:00,  2.78s/it]
    

    Completed SHAP analysis:
      Shape: (100, 8)
      Mean additivity error: 1.39e-06
      Max additivity error: 5.13e-06
    
    ### Group 1: Queen Background ###
    
    Running SHAP analysis on 100 selected samples...
    Configuration: queen background, decision_aligned mode
    Expected time: ~300 seconds
    

    Computing SHAP: 100%|████████████████████████████████████████████████████████████████| 100/100 [04:47<00:00,  2.87s/it]
    

    Completed SHAP analysis:
      Shape: (100, 8)
      Mean additivity error: 1.28e-06
      Max additivity error: 7.64e-06
    
    ======================================================================
    RUNNING FULL SHAP ANALYSIS - GROUP 2
    ======================================================================
    
    ### Group 2: Mean Background ###
    
    Running SHAP analysis on 100 selected samples...
    Configuration: mean background, decision_aligned mode
    Expected time: ~300 seconds
    

    Computing SHAP: 100%|████████████████████████████████████████████████████████████████| 100/100 [07:18<00:00,  4.38s/it]
    

    Completed SHAP analysis:
      Shape: (100, 10)
      Mean additivity error: 9.64e-07
      Max additivity error: 5.01e-06
    
    ### Group 2: Queen Background ###
    
    Running SHAP analysis on 100 selected samples...
    Configuration: queen background, decision_aligned mode
    Expected time: ~300 seconds
    

    Computing SHAP: 100%|████████████████████████████████████████████████████████████████| 100/100 [07:31<00:00,  4.51s/it]

    Completed SHAP analysis:
      Shape: (100, 10)
      Mean additivity error: 1.01e-06
      Max additivity error: 5.68e-06
    
    ======================================================================
    FULL SHAP ANALYSIS COMPLETE FOR BOTH GROUPS
    ======================================================================
    

    
    


```python
def compare_full_shap_configurations(shap_results_config1, shap_results_config2):
    """
    Compare two full SHAP configurations across all 100 samples
    
    Returns mean correlation, RMSE, etc. across all ROIs
    """
    correlations = []
    rmses = []
    
    for i in range(len(shap_results_config1['shap_values'])):
        shap1 = shap_results_config1['shap_values'][i]
        shap2 = shap_results_config2['shap_values'][i]
        
        corr = np.corrcoef(shap1, shap2)[0, 1]
        rmse = np.sqrt(np.mean((shap1 - shap2) ** 2))
        
        correlations.append(corr)
        rmses.append(rmse)
    
    return {
        'mean_correlation': np.mean(correlations),
        'mean_rmse': np.mean(rmses),
        'per_sample_correlations': correlations,
        'per_sample_rmses': rmses
    }
```

## DCG Comparison Analysis


```python
class DCGComparisonFramework:
    """
    Framework for comparing SHAP variable importance rankings with MLP gradient rankings
    using normalized Discounted Cumulative Gain (nDCG)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def rank_features_by_importance(self, importance_values, handle_ties='random'):
        """
        Rank features by importance values (higher importance = better rank)
        
        Parameters:
        - importance_values: Array of importance scores
        - handle_ties: 'random' to randomly break ties
        
        Returns:
        - rankings: Array where rankings[i] is the rank of feature i (1-based, 1=most important)
        """
        if handle_ties == 'random':
            # Add small random noise to break ties
            noise = np.random.random(len(importance_values)) * 1e-10
            noisy_values = importance_values + noise
            # Get ranks (argsort gives indices, we want ranks)
            sorted_indices = np.argsort(-noisy_values)  # Negative for descending order
            rankings = np.empty_like(sorted_indices)
            rankings[sorted_indices] = np.arange(1, len(importance_values) + 1)
        else:
            # Standard ranking (ties get same rank)
            rankings = stats.rankdata(-importance_values, method='average')
        
        return rankings.astype(int)
    
    def calculate_dcg(self, relevance_scores, k=None):
        """
        Calculate Discounted Cumulative Gain
        
        Parameters:
        - relevance_scores: Array of relevance scores in ranked order
        - k: Calculate DCG@k (None for full length)
        
        Returns:
        - dcg: DCG score
        """
        if k is not None:
            relevance_scores = relevance_scores[:k]
        
        if len(relevance_scores) == 0:
            return 0.0
        
        # DCG = rel_1 + sum(rel_i / log2(i+1)) for i=2 to k
        dcg = relevance_scores[0]
        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / np.log2(i + 2)  # i+2 because we start from i=1
        
        return dcg
    
    def calculate_ndcg(self, predicted_ranking, true_relevance, k=None):
        """
        Calculate normalized DCG comparing predicted ranking against true relevance
        
        Parameters:
        - predicted_ranking: Rankings from SHAP (1-based ranks)
        - true_relevance: True importance scores from MLP gradients
        - k: Calculate nDCG@k (None for full length)
        
        Returns:
        - ndcg: Normalized DCG score (0-1, higher is better)
        """
        n_features = len(predicted_ranking)
        if k is None:
            k = n_features
        
        # Create relevance scores based on true importance
        # Higher true importance = higher relevance
        max_true = np.max(np.abs(true_relevance))
        if max_true == 0:
            return 0.0  # All features have zero importance
        
        # Normalize true relevance to [0, 1] range for relevance scores
        normalized_relevance = np.abs(true_relevance) / max_true
        
        # Get relevance scores in the order of predicted ranking
        # predicted_ranking is 1-based, convert to 0-based indices
        ranking_indices = np.argsort(predicted_ranking)  # Features sorted by their rank
        relevance_in_predicted_order = normalized_relevance[ranking_indices]
        
        # Calculate DCG for predicted ranking
        dcg = self.calculate_dcg(relevance_in_predicted_order, k)
        
        # Calculate ideal DCG (best possible ranking)
        ideal_relevance_order = np.sort(normalized_relevance)[::-1]  # Descending order
        ideal_dcg = self.calculate_dcg(ideal_relevance_order, k)
        
        # Calculate nDCG
        if ideal_dcg == 0:
            return 0.0
        
        ndcg = dcg / ideal_dcg
        return ndcg
    
    def compare_shap_vs_mlp_for_sample(self, shap_values, mlp_importance, k=None):
        """
        Compare SHAP vs MLP importance for a single sample
        
        Parameters:
        - shap_values: SHAP importance values
        - mlp_importance: MLP gradient importance values
        - k: Calculate nDCG@k (None for all features)
        
        Returns:
        - ndcg_score: nDCG score
        - shap_ranking: SHAP feature rankings
        - mlp_ranking: MLP feature rankings (for reference)
        """
        # Rank features by absolute importance
        shap_ranking = self.rank_features_by_importance(np.abs(shap_values))
        mlp_ranking = self.rank_features_by_importance(np.abs(mlp_importance))
        
        # Calculate nDCG using SHAP ranking vs MLP importance
        ndcg_score = self.calculate_ndcg(shap_ranking, mlp_importance, k)
        
        return {
            'ndcg_score': ndcg_score,
            'shap_ranking': shap_ranking,
            'mlp_ranking': mlp_ranking,
            'shap_values': shap_values,
            'mlp_importance': mlp_importance
        }
    
    def run_full_dcg_analysis(self, shap_results, analysis_data, dataset_name='smooth_gradient', k=None):
        """
        Run complete DCG analysis for SHAP vs MLP comparisons
        
        Parameters:
        - shap_results: Dict with SHAP results from run_full_shap_analysis_on_selected_samples
        - analysis_data: Results from setup_consistent_samples_for_analysis (full dict)
        - dataset_name: Which dataset to analyze (default: 'smooth_gradient')
        - k: Calculate nDCG@k (None for all features)
        
        Returns:
        - dcg_results: Comprehensive DCG analysis results
        """
        # Get the specific dataset
        if dataset_name not in analysis_data:
            raise ValueError(f"Dataset {dataset_name} not found in analysis_data. Available: {list(analysis_data.keys())}")
        
        mlp_importance = analysis_data[dataset_name]['gradient_reference_importance']
        shap_values = shap_results['shap_values']
        selected_samples = shap_results['selected_samples']
        configuration = shap_results['configuration']
        
        print(f"Running DCG analysis for {configuration[0]} background + {configuration[1]} mode")
        print(f"Comparing {len(shap_values)} samples...")
        
        # Store results for each sample
        sample_results = []
        ndcg_scores = []
        
        for i in tqdm(range(len(shap_values)), desc="Computing nDCG scores"):
            comparison = self.compare_shap_vs_mlp_for_sample(
                shap_values[i], mlp_importance[i], k
            )
            sample_results.append(comparison)
            ndcg_scores.append(comparison['ndcg_score'])
        
        ndcg_array = np.array(ndcg_scores)
        
        # Create spatial grid for visualization
        grid_size = int(np.sqrt(len(ndcg_scores)))
        ndcg_grid = ndcg_array.reshape(grid_size, grid_size)
        
        results = {
            'configuration': configuration,
            'ndcg_scores': ndcg_array,
            'ndcg_grid': ndcg_grid,
            'sample_results': sample_results,
            'selected_samples': selected_samples,
            'statistics': {
                'mean_ndcg': np.mean(ndcg_array),
                'std_ndcg': np.std(ndcg_array),
                'min_ndcg': np.min(ndcg_array),
                'max_ndcg': np.max(ndcg_array),
                'median_ndcg': np.median(ndcg_array),
                'q25_ndcg': np.percentile(ndcg_array, 25),
                'q75_ndcg': np.percentile(ndcg_array, 75)
            },
            'k': k if k is not None else len(shap_values[0])
        }
        
        print(f"DCG Analysis Results:")
        print(f"  Mean nDCG: {results['statistics']['mean_ndcg']:.4f} ± {results['statistics']['std_ndcg']:.4f}")
        print(f"  Range: [{results['statistics']['min_ndcg']:.4f}, {results['statistics']['max_ndcg']:.4f}]")
        print(f"  Median: {results['statistics']['median_ndcg']:.4f}")
        
        return results
    
    def visualize_dcg_spatial_pattern(self, dcg_results, title_prefix="", figsize=(10, 8)):
        """
        Visualize spatial pattern of nDCG scores
        
        Parameters:
        - dcg_results: Results from run_full_dcg_analysis
        - title_prefix: Prefix for plot title
        - figsize: Figure size
        """
        config = dcg_results['configuration']
        ndcg_grid = dcg_results['ndcg_grid']
        stats = dcg_results['statistics']
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        im = plt.imshow(ndcg_grid, cmap='viridis', vmin=0, vmax=1, origin='lower')
        plt.colorbar(im, label='nDCG Score', shrink=0.8)
        
        # Title and labels
        config_str = f"{config[0].title()} Background + {config[1].replace('_', ' ').title()}"
        plt.title(f'{title_prefix}nDCG Scores: SHAP vs MLP\n{config_str}\n'
                 f'Mean: {stats["mean_ndcg"]:.3f} ± {stats["std_ndcg"]:.3f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Grid Column', fontsize=12)
        plt.ylabel('Grid Row', fontsize=12)
        
        # Add grid lines
        grid_size = ndcg_grid.shape[0]
        for i in range(grid_size + 1):
            plt.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
            plt.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        
        # Add text annotations for values
        for i in range(grid_size):
            for j in range(grid_size):
                value = ndcg_grid[i, j]
                text_color = 'white' if value < 0.5 else 'black'
                plt.text(j, i, f'{value:.3f}', ha='center', va='center',
                        color=text_color, fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return plt.gcf()
    
    def compare_configurations_statistically(self, dcg_results_list):
        """
        Statistical comparison between different SHAP configurations
        
        Parameters:
        - dcg_results_list: List of DCG results from different configurations
        
        Returns:
        - statistical_comparison: Dict with test results
        """
        print("Statistical Comparison of nDCG Scores Across Configurations")
        print("=" * 60)
        
        # Extract nDCG scores for each configuration
        config_scores = {}
        config_names = []
        
        for dcg_result in dcg_results_list:
            config = dcg_result['configuration']
            config_name = f"{config[0]}_{config[1]}"
            config_names.append(config_name)
            config_scores[config_name] = dcg_result['ndcg_scores']
        
        # Summary statistics
        print("Configuration Summary:")
        for name, scores in config_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {name}: {mean_score:.4f} ± {std_score:.4f}")
        
        # Overall ANOVA-style test (Friedman test for related samples)
        if len(config_scores) > 2:
            scores_array = np.array(list(config_scores.values()))
            friedman_stat, friedman_p = friedmanchisquare(*scores_array)
            print(f"\nFriedman Test (overall difference):")
            print(f"  Statistic: {friedman_stat:.4f}")
            print(f"  p-value: {friedman_p:.4f}")
            print(f"  Significant: {'Yes' if friedman_p < 0.05 else 'No'}")
        
        # Pairwise comparisons using Wilcoxon signed-rank test
        print(f"\nPairwise Comparisons (Wilcoxon Signed-Rank Test):")
        n_comparisons = len(config_names) * (len(config_names) - 1) // 2  # Add this line
        pairwise_results = {}
        
        for i, name1 in enumerate(config_names):
            for j, name2 in enumerate(config_names):
                if i < j:  # Avoid duplicate comparisons
                    scores1 = config_scores[name1]
                    scores2 = config_scores[name2]
                    
                    # Wilcoxon signed-rank test with error handling
                    try:
                        statistic, p_value = wilcoxon(scores1, scores2, alternative='two-sided')
                    except (ValueError, RuntimeWarning) as e:
                        # Handle case where differences are all zero or very small
                        print(f"    Warning: Wilcoxon test failed for {name1} vs {name2}: {str(e)}")
                        statistic = 0.0
                        p_value = 1.0  # No difference

                    # Paired t-test as additional comparison
                    from scipy.stats import ttest_rel
                    t_statistic, t_p_value = ttest_rel(scores1, scores2)
                    t_bonferroni_p = min(t_p_value * n_comparisons, 1.0)
                    
                    # Effect size (difference in means)
                    effect_size = np.mean(scores1) - np.mean(scores2)
                    
                    # Bonferroni correction
                    bonferroni_p = min(p_value * n_comparisons, 1.0)
                    
                    pairwise_results[f"{name1}_vs_{name2}"] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'bonferroni_p': bonferroni_p,
                        'effect_size': effect_size,
                        'significant': bonferroni_p < 0.05
                    }
                    
                    significance = "***" if bonferroni_p < 0.001 else "**" if bonferroni_p < 0.01 else "*" if bonferroni_p < 0.05 else "ns"
                    print(f"  {name1} vs {name2}:")
                    print(f"    Difference: {effect_size:+.4f}")
                    print(f"    p-value: {p_value:.4f} (Bonferroni: {bonferroni_p:.4f}) {significance}")
        
        return {
            'config_names': config_names,
            'config_scores': config_scores,
            'friedman_test': {
                'statistic': friedman_stat if len(config_scores) > 2 else None,
                'p_value': friedman_p if len(config_scores) > 2 else None
            } if len(config_scores) > 2 else None,
            'pairwise_comparisons': pairwise_results
        }
    
    def create_comparison_summary_plot(self, dcg_results_list, figsize=(15, 10)):
        """
        Create a comprehensive comparison plot for all configurations
        
        Parameters:
        - dcg_results_list: List of DCG results from different configurations
        - figsize: Figure size
        """
        n_configs = len(dcg_results_list)
        
        fig, axes = plt.subplots(2, n_configs, figsize=figsize)
        if n_configs == 1:
            axes = axes.reshape(2, 1)
        
        # Extract data for box plots
        all_scores = []
        config_labels = []
        
        for i, dcg_result in enumerate(dcg_results_list):
            config = dcg_result['configuration']
            config_name = f"{config[0].title()}\n{config[1].replace('_', ' ').title()}"
            config_labels.append(config_name)
            all_scores.append(dcg_result['ndcg_scores'])
            
            # Spatial heatmap (top row)
            ax1 = axes[0, i]
            ndcg_grid = dcg_result['ndcg_grid']
            im = ax1.imshow(ndcg_grid, cmap='viridis', vmin=0, vmax=1, origin='lower')
            ax1.set_title(f'{config_name}\nMean: {dcg_result["statistics"]["mean_ndcg"]:.3f}')
            ax1.set_xlabel('Column')
            ax1.set_ylabel('Row')
            
            # Add colorbar to rightmost plot
            if i == n_configs - 1:
                cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
                cbar.set_label('nDCG Score')
        
        # Box plot comparison (bottom row, spanning all columns)
        ax2 = plt.subplot(2, 1, 2)
        box_plot = ax2.boxplot(all_scores, tick_labels=config_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']
        for patch, color in zip(box_plot['boxes'], colors[:n_configs]):
            patch.set_facecolor(color)
        
        ax2.set_ylabel('nDCG Score')
        ax2.set_title('Distribution of nDCG Scores Across Configurations')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add mean values as points
        means = [np.mean(scores) for scores in all_scores]
        ax2.scatter(range(1, n_configs + 1), means, color='red', s=50, zorder=3, label='Mean')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig
```


```python
print("\n" + "="*70)
print("DCG COMPARISON: SHAP vs MLP GRADIENTS")
print("="*70)

# Initialize DCG framework
dcg_framework = DCGComparisonFramework(random_state=42)

# Group 1: Mean background
print("\n### Group 1: Mean Background ###")
dcg_results_mean_g1 = dcg_framework.run_full_dcg_analysis(
    shap_results=shap_results_mean_g1,
    analysis_data=analysis_data_group1,
    dataset_name='smooth_gradient'
)

# Group 1: Queen background
print("\n### Group 1: Queen Background ###")
dcg_results_queen_g1 = dcg_framework.run_full_dcg_analysis(
    shap_results=shap_results_queen_g1,
    analysis_data=analysis_data_group1
)

# Visualize spatial patterns
dcg_framework.visualize_dcg_spatial_pattern(dcg_results_mean_g1, title_prefix="Group 1: ")
dcg_framework.visualize_dcg_spatial_pattern(dcg_results_queen_g1, title_prefix="Group 1: ")

# Statistical comparison
print("\n### Comparing Mean vs Queen for Group 1 ###")
comparison_stats_g1 = dcg_framework.compare_configurations_statistically(
    [dcg_results_mean_g1, dcg_results_queen_g1]
)
```

    
    ======================================================================
    DCG COMPARISON: SHAP vs MLP GRADIENTS
    ======================================================================
    
    ### Group 1: Mean Background ###
    


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    Cell In[123], line 10
          8 # Group 1: Mean background
          9 print("\n### Group 1: Mean Background ###")
    ---> 10 dcg_results_mean_g1 = dcg_framework.run_full_dcg_analysis(
         11     shap_results=shap_results_mean_g1,
         12     analysis_data=analysis_data_group1,
         13     dataset_name='smooth_gradient'
         14 )
         16 # Group 1: Queen background
         17 print("\n### Group 1: Queen Background ###")
    

    Cell In[120], line 151, in DCGComparisonFramework.run_full_dcg_analysis(self, shap_results, analysis_data, dataset_name, k)
        148     raise ValueError(f"Dataset {dataset_name} not found in analysis_data. Available: {list(analysis_data.keys())}")
        150 mlp_importance = analysis_data[dataset_name]['gradient_reference_importance']
    --> 151 shap_values = shap_results['shap_values']
        152 selected_samples = shap_results['selected_samples']
        153 configuration = shap_results['configuration']
    

    TypeError: 'NoneType' object is not subscriptable



```python
# Check what the SHAP results contain
print("Type of shap_results_mean_g1:", type(shap_results_mean_g1))
print("Contents:", shap_results_mean_g1)
```


```python

```
