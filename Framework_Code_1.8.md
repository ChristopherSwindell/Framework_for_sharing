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
