# Single Histogram Filter

Single Histogram Filter (*SHF* in short for the following context), is a tool introduced in [v0.6.0](https://github.com/stephen-hqxu/superterrainplus/releases/tag/v0.6.0) designed for filtering any image with discrete pixel format. SHF generates an analytical histogram for every pixel within a given radius on the image, each histogram bin is labelled by the pixel value, and stores the frequency of appearance of each pixel in the filter kernel.

> The following documentation gives a brief summary to SHF, you may want to read the original paper for full details. If you are using SHF in any published research, please cite the work.

## Motivation and Principle

SHF was first designed to address the problem of generating a multi-biome heightfield terrain. This is achieved by assigning different generator parameters for biomes, and each biome is uniquely identified by its biome ID. The biomemap is a texture acting as a lookup table for biome ID.

This solution will most likely lead to discontinuous terrain, resulting in sharp cliff caused by this discontinuity between biomes, because each heightfield generator lookups biome ID from the current pixel without knowing the biome around it, so that they are all working independent of each other.

Solutions exist for modern heightfield-based terrain generators with multi-biome features. One of the most notable example, *Minecraft*, makes use of subsampling of heightmap, to interpolate or smooth out the sharp edges when super-sampling from the subsampled heightmap. Another popular solution attempts this from another perspective with use of low-pass filter such as Gaussian blur filter.

The aforementioned approaches work in principle and demonstrate successful results for most voxel-based terrain in practice, but fail to provide consistent and convincing result for high quality mesh-based terrain. Subsampling abandons details on the heightmap and leads to interpolation artefacts when super-sampled; whereas low-pass filter blurs out both sharp cliff areas as well as fractal noise and erosion details created through the initial heightmap synthesis and simulations.

This shows us removing the biome transition artefacts after the heightmap is generated can be unreliable, we can instead avoiding this artefact completely before the generation starts. Biomemap itself is a special texture, differs from heightmap, each pixel value is discrete, such that applying regular filter and producing continuos values such as 2.5 will not be sensible.

By knowing the weight factors of each biome at a pixel within a range of neighbour pixels, heightfield generator can understand how much each biome should contribute to generate the height value for the current pixel. The final height at a particular pixel can therefore be calculated as the weighted sum over height values as if the pixel is generated only for one biome.

$$
\sum_{b = 0}^{N_{biome}} w(b)h(b)
$$

where $b$ is the biome ID, $w(b)$ is the weight factor of biome $b$ and $h(b)$ is the height at the current pixel in biome $b$. Given the following conditions always hold:

$$
\forall b, \sum_{b = 0}^{N_{biome}} w(b) = 1 \text{ and } w(b), h(b) \in [0, 1]
$$

## Data Structure

Execution of SHF yields an array of single histogram, each pixel will have exactly one single histogram; The single histogram is an array of flattened bins, each bin contains a label denoting the responsible biome ID and the weight of this bin. The single histogram for a particular pixel can be looked up an array of histogram start offset, using the index of pixel.

Suppose the index of the current pixel is $i$, the array of bins is $Bin$ and the array of histogram start offset is $HSO$. The begin and end iterator pair of the range of bins for the current pixel can be calculated as:

$$
[Bin + HSO[i], Bin + HSO[i + 1])
$$

Note that following standard C++ convention, the range is half opened. For this purpose, such that the array index $i + 1$ into $HSO$ at the last pixel is valid, an additional offset denoting the total number of bin is inserted at the end.

## Performance

As of current, SHF is implemented on CPU, and make use of 2 major filter optimisations to achieve an asymptotic runtime complexity of constant with respect to filter kernel radius.

The performance can be greatly improved by exploiting the fact that SHF is in fact separable. A separable filter allows breaking down of a 2D filter into two 1D filters, effectively reducing the time complexity from quadratic to linear w.r.t radius of kernel.

Accumulation is commonly used in filter with equal kernel components, meaning a uniform filter function is applied to all pixels. Instead of discarding the old kernel and recompute the histogram for the next kernel, the new kernel $K_{h}[y, x + 1]$ can be modified from the old kernel $K_{h}[y, x]$ by removing the left-most pixel in the old kernel position, $x - r$, and adding the right-most pixel in the new kernel position, $x + r + 1$.

$$
K_{h}[y, x + 1] = K_{h}[y, x] - f[y, x - r] + f[y, x + r + 1]
$$

where $f_{h}$ is the filter function applied to the pixel during the horizontal pass. The same principle can be applied when performing the vertical pass.

Finally, the implementation is parallelised on 4 CPU threads using for-loop block parallelism. The overall performance comparison with different optimisation techniques applied can be found in the following table:

| Platform | Brute Force | Separable Kernel | Separable Accumulation |
| -------- | ----------- | ---------------- | ---------------------- |
| Time Complexity | $O(Nr^{2})$ | $O(Nr)$ | $O(N + r)$ |
| Single CPU | 1420 | --- | --- |
| 4-thread CPU | 780 | 250 | 25 |
| GPU | 510 | 270 | --- |

> Times are measured in *ms* using mean runtime of 10 executions. Test program was compiled by MSVC 16.11, CUDA 11.3, if applicable. Compiler optimisation is turned on to max favouring speed. The testing resolution of biomemap is 512 x 512, and biome IDs are distributed uniformly random in range [0, 15), with kernel radius of 32.