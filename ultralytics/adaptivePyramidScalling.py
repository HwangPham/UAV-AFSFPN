# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:06:12 2025

@author: Z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class APS_FPN(nn.Module):
    def __init__(self, in_channels, out_channels, pyramid_levels=5):
        """
        Initializes the Adaptive Pyramid Scaling Feature Pyramid Network (APS-FPN).
        
        Args:
            in_channels: Number of input channels (e.g., from the backbone).
            out_channels: Number of output channels (final feature map channels).
            pyramid_levels: Number of levels in the feature pyramid.
        """
        super(APS_FPN, self).__init__()
        
        self.pyramid_levels = pyramid_levels
        
        # Define the lateral convolutions for each level in the pyramid
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(pyramid_levels)
        ])
        
        # Define top-down convolutions to upsample and merge features
        self.topdown_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in range(pyramid_levels - 1)
        ])
        
    def forward(self, feature_maps, object_sizes):
        """
        Forward pass through the Adaptive Pyramid Scaling FPN.

        Args:
            feature_maps: List of feature maps from the backbone at different levels (e.g., [C2, C3, C4, C5]).
            object_sizes: List of object sizes (height, width) of objects in the image. This can be used to adjust scaling dynamically.

        Returns:
            List of feature maps at different pyramid levels after applying APS.
        """
        
        # Apply lateral convolutions to each feature map to reduce dimensions
        laterals = [lateral_conv(fm) for lateral_conv, fm in zip(self.lateral_convs, feature_maps)]

        # Create a top-down path for FPN feature pyramid construction
        pyramids = [laterals[-1]]

        # Perform top-down feature scaling and adjust based on object sizes
        for i in range(self.pyramid_levels - 2, -1, -1):
            # Use the size of the current lateral feature map as the target size
            target_size = laterals[i].shape[2:]  # (height, width)

            # Upsample to match the size of the current lateral feature map
            upsampled_feature_map = F.interpolate(
                pyramids[-1],
                size=target_size,  # Use explicit size to match lateral feature map
                mode='bilinear',
                align_corners=False
            )

            # Merge the upsampled feature map with the current lateral feature map
            merged_feature_map = laterals[i] + upsampled_feature_map

            # Apply a 3x3 convolution to refine the feature map
            pyramids.append(self.topdown_convs[i](merged_feature_map))

        # Reverse pyramids to maintain the original top-down order
        pyramids = pyramids[::-1]

        return pyramids
    
    def calculate_scale_factor(self, object_size):
        """
        Calculate the scale factor dynamically based on the size of the objects in the feature map.
        
        Args:
            object_size: A tuple (height, width) of the object size in the current feature map.
        
        Returns:
            scale_factor: A dynamically calculated scale factor for the feature map.
        """
        # Example: the scale factor could be based on the ratio of the object size to the feature map size
        height, width = object_size
        scale_factor = (height * width) ** 0.5 / 256  # Simple example: scale based on object area size
        min_scale_factor = 0.000001
        return max(scale_factor, min_scale_factor)


# Example Usage:
# Define the model with input channels (e.g., from a ResNet backbone) and output channels for the FPN
in_channels = 256
out_channels = 256
model = APS_FPN(in_channels, out_channels, pyramid_levels=5)

# Example feature maps from a backbone network (e.g., ResNet), each with different sizes
# These are just example tensor shapes for illustration
feature_maps = [torch.randn(1, in_channels, h, w) for h, w in [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]]

# Example object sizes in the image (height, width) for adaptive scaling
object_sizes = [(20, 20), (30, 30), (40, 40), (50, 50), (60, 60)]

# Pass the feature maps and object sizes through the APS-FPN
output_pyramids = model(feature_maps, object_sizes)

# Each element in `output_pyramids` corresponds to a feature map at a different level of the pyramid
for idx, pyramid_level in enumerate(output_pyramids):
    print(f"Pyramid Level {idx+1}: {pyramid_level.shape}")