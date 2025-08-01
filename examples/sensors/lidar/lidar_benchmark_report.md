# Genesis LiDAR Sensor Performance Benchmark Report

Generated on: 2025-08-01 18:40:44

## Executive Summary

- **Total configurations tested**: 38
- **Performance range**: 3.44 - 22839.18 ms
- **Efficiency range**: 108 - 13250 rays/ms
- **Best configuration**: 1 envs × 64 lines × 32 obstacles

## Detailed Results

### Top 10 Fastest Configurations

|   n_envs |   n_scan_lines |   n_obstacles |   read_time_ms |   efficiency_rays_per_ms |
|---------:|---------------:|--------------:|---------------:|-------------------------:|
|     1.00 |          64.00 |         32.00 |           3.44 |                   595.92 |
|     1.00 |         512.00 |         32.00 |           3.60 |                  4554.09 |
|     1.00 |        2048.00 |         32.00 |           5.20 |                  9456.53 |
|     1.00 |        4096.00 |         32.00 |           6.98 |                  9393.41 |
|     1.00 |          64.00 |         96.00 |           8.64 |                   237.12 |
|     1.00 |         512.00 |         96.00 |           9.82 |                  1667.98 |
|     1.00 |        2048.00 |         96.00 |          16.76 |                  2931.90 |
|     1.00 |          64.00 |        256.00 |          18.94 |                   108.15 |
|     1.00 |         512.00 |        256.00 |          21.82 |                   750.93 |
|     1.00 |        4096.00 |         96.00 |          23.03 |                  2845.77 |

### Top 10 Most Efficient Configurations

|   n_envs |   n_scan_lines |   n_obstacles |   read_time_ms |   efficiency_rays_per_ms |
|---------:|---------------:|--------------:|---------------:|-------------------------:|
|   512.00 |          64.00 |         32.00 |          79.14 |                 13250.38 |
|   512.00 |         512.00 |         32.00 |         664.40 |                 12625.93 |
|  2048.00 |          64.00 |         32.00 |         333.26 |                 12585.54 |
|  4096.00 |          64.00 |         32.00 |         674.99 |                 12427.71 |
|  4096.00 |         512.00 |         32.00 |        1353.70 |                 12393.62 |
|  8192.00 |         512.00 |         32.00 |        2709.30 |                 12384.90 |
|   512.00 |        4096.00 |         32.00 |        2734.66 |                 12270.04 |
|  2048.00 |        2048.00 |         32.00 |        2736.49 |                 12261.86 |
|  8192.00 |          64.00 |         32.00 |        1370.24 |                 12244.03 |
|  2048.00 |         512.00 |         32.00 |        2748.36 |                 12208.90 |

## Scaling Analysis

### Environment Scaling

| Environments | Read Time (ms) | Parallel Efficiency |
|--------------|----------------|---------------------|
| 1 | 3.44 | 1.00x (baseline) |
| 512 | 79.14 | 22.24x |
| 2048 | 333.26 | 21.12x |
| 4096 | 674.99 | 20.85x |
| 8192 | 1370.24 | 20.55x |

## Recommendations

### For Best Performance
- Use fewer environments for latency-critical applications
- Balance scan lines vs points per line based on required resolution
- Consider obstacle density impact on performance

### For Best Throughput
- Use higher environment counts for batch processing
- Monitor memory usage with large configurations
- Consider adaptive ray count based on environment complexity

## Technical Details

### Correlation Analysis

| Factor | Correlation with Read Time |
|--------|----------------------------|
| Environments | 0.309 |
| Scan Lines | 0.102 |
| Obstacles | 0.293 |

