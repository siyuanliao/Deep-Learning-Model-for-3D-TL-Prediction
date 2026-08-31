import numpy as np
from scipy import ndimage

def replace_inf_with_local_mean(tensor, kernel_size=3, max_iterations=10):
    """
    Replace Inf values in a 4-D tensor using the mean of neighboring finite values.
    
    Args:
        tensor: 4-D NumPy array.
        kernel_size: Neighborhood size. Must be odd. Default is 3.
        max_iterations: Maximum iterations for handling contiguous Inf regions.
    
    Returns:
        The processed 4-D tensor.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    # Create a floating-point copy of the input tensor.
    result = tensor.copy().astype(np.float64)
    
    # Build a mask for Inf values.
    inf_mask = np.isinf(result)
    
    # Return immediately when no Inf values are present.
    if not np.any(inf_mask):
        return result
    
    # Iterate until all Inf values are replaced or the iteration limit is reached.
    for iteration in range(max_iterations):
        # Check whether any Inf values remain.
        current_inf_mask = np.isinf(result)
        if not np.any(current_inf_mask):
            break
            
        # Replace Inf with NaN temporarily so neighborhood statistics can ignore them.
        temp = result.copy()
        temp[current_inf_mask] = np.nan
        
        # Process each batch and channel independently.
        for b in range(temp.shape[0]):
            for c in range(temp.shape[1]):
                # Extract the current 2-D channel slice.
                channel_slice = temp[b, c, :, :]
                
                # Apply a local mean filter while ignoring NaN values.
                mean_filtered = ndimage.generic_filter(
                    channel_slice, 
                    np.nanmean, 
                    size=kernel_size, 
                    mode='constant', 
                    cval=np.nan
                )
                
                # Locate Inf values in the current channel.
                channel_inf_mask = np.isinf(result[b, c, :, :])
                
                # Replace Inf values with the mean of neighboring finite values.
                if np.any(channel_inf_mask):
                    # Replace only when at least one finite neighbor is available.
                    valid_replacements = ~np.isnan(mean_filtered[channel_inf_mask])
                    replace_indices = np.where(channel_inf_mask)
                    valid_indices = (replace_indices[0][valid_replacements], 
                                   replace_indices[1][valid_replacements])
                    
                    if len(valid_indices[0]) > 0:
                        result[b, c][valid_indices] = mean_filtered[valid_indices]
        
        print(f"Iteration {iteration + 1}: remaining Inf values = {np.sum(np.isinf(result))}")
    
    # Replace any remaining Inf values with the global mean of finite values.
    final_inf_mask = np.isinf(result)
    if np.any(final_inf_mask):
        global_mean = np.nanmean(temp)  # Inf values in temp have already been converted to NaN.
        if not np.isnan(global_mean):
            result[final_inf_mask] = global_mean
        else:
            # Use zero only if the entire tensor contains Inf values.
            result[final_inf_mask] = 0.0
        print(f"Replaced the remaining {np.sum(final_inf_mask)} Inf values with the global mean")
    
    return result

def replace_inf_with_expanding_mean(tensor, max_window_size=11):
    """
    Replace Inf values using an expanding neighborhood until finite values are found.
    
    Args:
        tensor: 4-D NumPy array.
        max_window_size: Maximum neighborhood size. Must be odd.
    
    Returns:
        The processed 4-D tensor.
    """
    print("ok")
    result = tensor.copy().astype(np.float64)
    inf_mask = np.isinf(result)
    
    if not np.any(inf_mask):
        return result
    
    # Get the coordinates of all Inf values.
    inf_coords = np.where(inf_mask)
    
    for idx in range(len(inf_coords[0])):
        print(f"Progress: {idx}/{len(inf_coords[0])}")
        b, c, h, w = inf_coords[0][idx], inf_coords[1][idx], inf_coords[2][idx], inf_coords[3][idx]
        
        # Increase the window size until finite neighbors are found.
        for window_size in range(3, max_window_size + 1, 2):
            half_window = window_size // 2
            
            # Compute window boundaries.
            h_start = max(0, h - half_window)
            h_end = min(result.shape[2], h + half_window + 1)
            w_start = max(0, w - half_window)
            w_end = min(result.shape[3], w + half_window + 1)
            
            # Extract the current neighborhood.
            window = result[b, c, h_start:h_end, w_start:w_end]
            
            # Collect finite values in the neighborhood.
            non_inf_values = window[~np.isinf(window)]
            
            if len(non_inf_values) > 0:
                # Replace the current Inf value with the neighborhood mean.
                result[b, c, h, w] = np.mean(non_inf_values)
                break
        else:
            # Fall back to the global finite-value mean if no window contains valid values.
            non_inf_global = result[~np.isinf(result)]
            if len(non_inf_global) > 0:
                result[b, c, h, w] = np.mean(non_inf_global)
            else:
                result[b, c, h, w] = 0.0
    
    return result

# Test utilities
def test_inf_replacement():
    """Run a basic test of the Inf-replacement utilities."""
    # Create a 4-D tensor containing several Inf values.
    np.random.seed(42)
    tensor = np.random.randn(2, 3, 5, 5).astype(np.float32)
    
    # Add isolated and contiguous Inf regions.
    tensor[0, 0, 1:4, 1:4] = np.inf  # 3 x 3 Inf region
    tensor[0, 1, 2, 2] = np.inf      # isolated Inf
    tensor[1, 2, 0, 0] = -np.inf     # negative Inf
    
    print("Number of Inf values in the original tensor:", np.sum(np.isinf(tensor)))
    print("Original tensor shape:", tensor.shape)
    
    # Method 1: local-mean replacement.
    result1 = replace_inf_with_local_mean(tensor)
    print("Inf values after method 1:", np.sum(np.isinf(result1)))
    
    # Method 2: expanding-window replacement.
    result2 = replace_inf_with_expanding_mean(tensor)
    print("Inf values after method 2:", np.sum(np.isinf(result2)))
    
    return result1, result2

def calculate_model_complexity(model):
    """
    Calculate total and trainable model parameter counts.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model complexity analysis")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Approximately {total_params/1e6:.2f} million parameters")
    
    # Reference parameter counts for scale comparison.
    print("\nReference model comparison:")
    print("LeNet-5: ~60,000 parameters")
    print("AlexNet: ~60 million parameters") 
    print(f"Current network: ~{total_params/1e6:.1f} million parameters")
    print(f"Parameter count relative to LeNet-5: {total_params/60000:.1f}x")
    
    return total_params, trainable_params

if __name__ == "__main__":
    result1, result2 = test_inf_replacement()