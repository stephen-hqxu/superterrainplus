Hello and welcome to this presentation, my name is Stephen and today I am going to present you a very interesting topic in computer graphics which is procedural generation, and specifically we will be looking at procedural techniques for terrain generation, simulation and photorealistic-rendering.

# Presentation overview

Today's presentation will be consist of the following parts, I will give you a brief idea of what procedural generation is, then we will move on to my project, how I achieved procedural generation and manage the project. And will be concluded by some limitations and future works for the project.

# Introduction

## What is procedural generation

So what exactly is procedural generation? Procedural refers to some procedures, steps, more generally when we are talking about computer science, algorithms. We are looking for generation of data using algorithms with minimal to no human intervention. The piece of data can be anything from, visually, model, texture and effects; to acoustically, sound.

Procedural generation is cheap because manual production of such data is usually expensive and requires professional skills. If we look at the pipeline here the procedural generator only requires a very small set of initial condition and this helps us to save space. A procedural generator usually involves random number generation, meaning the output data is less predictable because no two objects in our world are exactly the same.

## Where is can be used

Maybe at this point procedural generation still sounds very unfamiliar to you. Let's take a look at some modern examples. In game industry one of the most notable examples is *Minecraft*, which features procedural terrain generation that allows player to explore a world endlessly without feeling repetitive. Other examples like *Left 4 Dead 2* uses a procedural manner to spawn zombies based on player performance. In movie industry, *The Lord of the Rings* was the first film to implement procedural crowd generation with physics and animation, and inspired later films such as *Avatar* and *World World Z*. Imagine a combat scene in a science-fiction movie that involves thousands of characters, and they are, flying in the air or stacked onto each other, using human actors is not only expensive but dangerous; and procedural generation comes to rescue.

I would like to divide procedural generation techniques into two categories, for games it is usually done in real-time and for movies with maximum realism. There exists a lot of different procedural engines but, surprisingly, there isn't a one-fits-all solution, meaning we cannot achieve performance and realism both at the same time due to the constraint of processing power of current hardware.

## Why it is necessary

I would like to ask myself, if we cannot have both at the same time, is there a way to develop a procedural engine that allows switching between performance mode for game development and realism mode for movies and arts, so it is an adaptive multi-purpose solution, with certain customisability?

# Project development

So today, I would like to introduce you **SueprTerrain+**, the procedural landscape generator with physics simulation and photorealistic-rendering. It comes with 4 independent modules, a terrain generator, a physics simulator, and diversity generator and finally the photorealistic-rendering engine.

## Overview

Unfortunately due to the limited development time I am not able to cover all topics in procedural generation. Instead we will be looking at some state-of-the-art techniques for model generation and texture synthesis, then we will move on the algorithmic physics simulation and finally procedural rendering. Right now we will start from real-time generation and rendering and shift towards realism. Don't worry if some of the terms sound unfamiliar to you, we will be going through them one by one shortly.

*Next slide*

The terrain engine will be implemented using C++, specifically I am using C++ 17. I wish I could use C++ 20 because there are a lot of amazing features but due to stability issues I would stick with the current choice. CUDA is the main platform for GPU computing because OpenCL is not very friendly to Nvidia GPU. For rendering with over 7 years of experience on OpenGL I prefer it over DirectX.

Things always become more complicated when GPU comes to play. To reduce the amount of work, I mainly work with Nvidia Turing architecture and so most optimisations are targeting this type of GPUs.

## Terrain generation

### The main approach

There are two main methods to generate a virtual world in a procedural manner. The most popular approach is by tessellating a flat plane and displace vertices using a height field, this is very fast and simple to implement but due to single-direction displacement no cave systems can be generated. A more realistic but also expensive solution is by using volumetric meshing. We will be focusing on the first method today.

*Next slide*

Here is a big picture of the displacement approach. We start from a flat plane and a height field with pixels representing displacement distance, after tessellation a terrain is obtained.

### Chunk

So how should we get a flat plane, and remember, the terrain needs to be infinite, which is not possible to achieve with finite amount of memory. The most popular way of doing this is by splitting the terrain into discrete units, called chunks, so that the computer can render each chunk on-demand. As the viewer moves, out-of-bound chunks are unloaded while new chunks are being generated; this process will be completed by the time viewer approaches to the edge of rendered chunks, to create an illusion of infinity.

By further discretising the chunk to allow better control towards the complexity of computation and mesh quality, a new method is introduced in the engine where chunk is broken down into a even smaller unit called tile.

### Noise

The next problem is to generate the height field. A common way is by using a noise. In this project a 2D perlin-simplex hybrid noise with my own improvement is used to allows better control to randomness.

*Next slide*

Here is a simple illustration of different types of noise. The rationale behind the choice of noise functions is, pattern. White noise is too random and I wouldn't think it is possible to extract any useful features from it. By contrast, pseudo-random noise gives us some nice patterns. The gradient noise already looks like a top-down view of a terrain, while cellular noise more resembles some sorts of cloud.

*Next slide*

Okay so now we got the infinite plane and a height field texture, let's put them together and see what we got in the engine.

The noise has done a pretty good job for simulating a terrain. In real-life, the terrain will undergo some deformations over time and hence a smooth mesh is inaccurate to model a realistic landscape. Another technique called multi-fractal noise which combines the results from many different noise functions, can be used to add some details. The screenshot shows two terrains at the same place, with and without fractal applied.

### Hydraulic processing

However, I am still not quite satisfy with that, because now the terrain starts to look a bit too erratic. In real-life, the landscape undergoes natural deformations such as earthquake and erosion. Is that possible to simulate that phenomenon, in real-time?

The answer is yes. One type of widely used physics simulation called hydraulic erosion can be used, which simulates how water erodes the terrain. Past researchers developed two types of erosion algorithms, being cell-based and particle-based. Cell-based approach is mainly applied to erosion with pre-defined permanent water source such as river and lake whereas particle-based simulates mainly rain drops. Although it seems like the cell-based approach is more suitable to be used in conjunction with high-performance computing techniques, the requirement of pre-definition is unfortunately not preferable nor easily achievable in this context.

*Next slide*

The algorithm is based on gradient descent. A water droplet is first spawned on the top of the terrain at a random location, then it starts its journey downhill. While on its trip, the droplet erodes the terrain, carries sediment from the ground. As it reaches the local minimum, it can no longer go up because the gravity is pulling it back. The water content is depleting and eventually evaporated, leaving all sediments it has carried behind.

This process is performed on the height field texture rather than the terrain mesh to reduce memory usage. As each water droplet is independent, a computing thread can be assigned to each droplet.

*Next slide*

The idea from the original algorithm is very simple. The result, well, shows that a copy-pasted version of algorithm does not work as intended. The reason behind is the engine is dealing with an infinite terrain, and the rain drop has no information about chunks outside the current boundary. Hence when it goes out-of-bound, its lifetime gets terminated immediately, leaving the area around the border to be eroded less frequently, and disconnected.

*Next slide*

Here's the improved version of the algorithm. Before start eroding the chunk, a range of neighbour chunks are defined, to create a free-slip range. This range of chunks are merged into a large chunk and rain drop is spawned only at the centre chunk.

You might ask, what if one neighbour is occupied by multiple erosion workers? This is solved by introducing a flag for each chunk, and the chunk will not be used as either the centre chunk or neighbour if it is currently being used. It guarantees that the access to all chunks are mutually exclusive.

*Next slide*

The major overhead in this new system mainly comes from merging neighbour chunks into a single buffer, from host to device. Here's my solution. When merging chunks, a simple linear memory copy can be used instead of doing a more expensive 2D matrix copy, experiment shows linear copy speed can usually reach the bandwidth of the system while 2D utilises less than a half. But now, array indices of each neighbour will be in *local index system* and hence the memory alignment will not be the same as how our algorithm, which works in *global index system*, is expecting. For example, suppose a droplet wants to travel from chunk 0 to chunk 1 from index 3 to 4 in global index; if we use this information directly in the memory, the droplet in fact travels to the next pixel in chunk 0, at index 4 in local index, which is equivalent to index 8 in global index. So what do we need to do is to somehow convert the pair (3,4) to (3,8).

But how? A global-local index table is then used to easily convert the index from global system to local system. Some simple proof shows that after conversion it still satisfies the requirement of coalesced memory access such that this extra memory read incurs negligible overhead.

To further reduce the number of times host communicate with device and maximise linear copy bandwidth, pinned memory is used as buffer and reduce the number of memory transaction to N number of neighbour to only one.

*Next slide*

Good, what about now? As shown border artefact has been eliminated successfully and area around the boundary looks more evenly eroded than it was done previously.

## Biome generation

### The classic way

At this point I am quite happy with the shape of the terrain. But it starts to feel boring because the mesh looks the same everywhere. I would like to introduce some variation or diversity, so called biome. Multi-biome combines different height field generators, allowing them to collaborate and produce a single height field texture.

How should we inform different height field generators so they know which pixel each of them is responsible for? Yes, just like height field, we can store this information in another map. A biome map defines a biome ID at each pixel, so that the engine can pick the correct generator based on the biome ID.

There are many different techniques to generate a biome map. Do you still remember our cellular noise, each cell looks like a region, then a biome can be defined within.

*Next slide*

Or maybe by using two noise texture, one defines the temperature and the other one defines precipitation, then a lookup table can be used to determine the biome based on these two values.

*Next slide*

Should I use them? After some considerations and experiments, despite they are simple to implement but come with some major disadvantages. With the use of noise, it is much more difficult to control each biome. What's worse, as the control to biome is lost, some biomes with extreme climate might be placed together, for example desert is next to tundra. Also there may exist some areas on the biome map that are not defined, such as the black areas on the cellular noise and empty areas on the climate graph. It will be nice as well to have some unique features for some biomes, like having an island at the middle of the ocean, and it is not cheap to perform edge detection.

### Grown Biome Algorithm

I finally decide to borrow this idea which they have been using in *Minecraft*. This algorithm works by randomly placing some sample points on an empty texture first. Then we are trying to grow these samples by performing a zoom operation, which enlarges the texture almost twice in size. For each 2x2 neighbouring numbers in the original array it produces a 3x3 array, where the corner values inherit from their corner counterparts of the original array, and the values in the middle get chosen randomly from their appropriate neighbours.

The original library is written in Java and only works on a single thread. With license agreement I ported it to C++ and improve it a bit so that it can be executed in parallel.

So now we have a biome map, the only thing left is assigning different generators based on biome ID. It should be simple, right?

### Multi-biome Heightfield Generation

It turns out to be, not so simple, I managed to create some unintended cliff. The problem arises when multiple heightfield generators are collaborating. Since each generators is independent therefore they have no knowledge about their surrounding. This can be fixed pretty easily by performing interpolation to ensure the biome edge transition is smooth.

*Next slide*

There are a lot of ways of doing interpolation, the simplest one is by generating a low-resolution heightfield and use bilinear interpolation to up-scale it. This is the approach used in *Minecraft* as well as many custom terrain generators, I tried this out as it is easy to implement however it makes my terrain looks even worse, creating some staircase effects. I guess this technique is more suitable for voxel terrain generation.

Another not-so-popular approach is by using image filters, such as Gaussian filters, to smooth the texture. While image filters are great, they come with two major drawbacks. It is being very slow when the filter radius is set to a large value and I am expected to use a pretty large kernel radius for a mesh terrain. In addition if a Gaussian filter is used for the whole texture it will not only smooth the edge but also other non-edge parts on the texture like the erosion details I just created. Although I can perform a filter pre-pass to determine where is the edge and apply filter only to the edge region, this may create a second discontinuity between the filtered and unfiltered areas. 

In general, both methods are not quite applicable in this scenario.

### Single Histogram Filter

I kind of like the idea of using an image filter because it can possibly retain most of the original details, but perhaps I need to derive one myself for this terrain generator to make sure it does the right thing.

If filtering the heightfield texture is not a good idea, how about trying to filter the biomemap which contains biome IDs. Filtering biomemap is not easy because it is a discrete format texture and it probably wouldn't make sense to have an ID of 1.5, for example.

Maybe I can borrow more ideas from other areas? And then I found shadow map which I will cover more later. Shadow map stores distance from light to any object, and it wouldn't make sense to interpolate either otherwise distance information will become invalid. There are many techniques to make the shadow smooth, one of them is PCF. It is done by finding the ratio of shadow region over non-shadow region in a filter kernel, and this ratio can be used as an intensity multiplier.

*Next slide*

PCF is only applicable for binary data, shadow region and non-shadow region. In biomemap, it is certain to have more than 2 biome IDs. With slight modification, a histogram-like data structure can be used, each bin can only hold one item, which is the biome ID; and the filter counts the number of each biome ID presented in the kernel and accumulate the count in the corresponding bin. Finally the ratio of each biome ID in the kernel can be found. This ratio can then be sent to multi-biome terrain generator and each generator knows how much they should contribute to the final output in a pixel, and so they can collaborate. The sum of ratio is always 1.

Stability is the biggest benefit of this filter. If there is only one biome presented in the kernel, the ratio of this biome will be evaluated as 1, and only the generator for this biome is used to write the heightfield data for this pixel.

But at what cost? In fact it is exceptionally expensive. The new filter carries the same time complexity as most 2D image filter, being big O of N times r square where N is the number of pixel and r is the kernel radius. That's not the end of it, there are many ways to improve it.

*Next slide*

The first optimisation is by exploiting the fact that this filter is separable. For a 2D separable filter, it can be broken down into two 1D filters in horizontal and vertical direction. This effectively reduces the number of operation, from r squared to 2r for every pixel. In this example, it has cut down the number of addition operation from 9, which is squared of 3; to 6, which is 3 plus 3.

*Next slide*

The second optimisation is by using cache, called accumulator. This technique was first developed for fast Gaussian filter, I found this very interesting and thought maybe I can integrate this into my filter as well.

An accumulator is just a variable containing results of the current filter kernel, in this filter, the accumulator is a histogram. At the beginning of the algorithm the accumulator is loaded.

*Next slide*

Normally when the working pixel advances, everything is discarded and the kernel will be loaded from scratch. We don't want that. Thanks to this formula, we can obtain the kernel for the next pixel by unloading the left-most pixel from the last kernel and loading the right-most pixel in the current kernel. This way, only 2 operations are needed regardless of radius of filter.

*Next silde*

How does both optimisations benefit us? In fact a lot. I implemented single histogram filter on both CPU and GPU using all three techniques. The brute-force convolution is undoubtedly the worst whereas the separable filter cuts down the CPU execution time by a half. GPU is a strange one here because probably, we are expecting something better from it, but this is not a surprise to me as this filter requires transferring large chunk of memory, the histogram, frequently, and memory operation is not what GPU is good at, so I don't think it is a good idea to carry GPU any further. With both optimisations, it only takes 60ms to finish.

*Next slide*

I am trying to squeeze more performance out of it, so I try to run this on another compiler. Surprisingly GCC performs twice as good as MSVC, more investigations show the slowness mainly comes from the data structure, which is built from an array list, in C++ it is called vector. I have no clue why vector implementation on MSVC is so slow so I decide to implement an array list myself. There is nothing special with my own implementation, I only keep functionalities that I need. Fortunately the result is rewarding, and the runtime has been cut down by over a half on MSVC.

By now, I have successfully reduced the runtime from over 1 second, to only 25ms, and it should suffice for just-in-time generation.

### Smoothed Multi-biome Heightfield Generation

With that being said, here is the result of a smoothed multi-biome terrain with single histogram filter applied. I also show here side-by-side comparisons of all previous stages of terrain generation.

## Texture splatting

After spent so many hours on shaping the terrain, I think it is a good time to move onto some rendering, at lease I start to get a bit tired with the blue, normal mapped terrain, time to get some real colour.

The technique for mapping texture onto a procedural mesh is called texture splatting, which involves use of a texture splatmap and it defines which texture should be used to colour a pixel.

The common usage of splatmap is by assigning a one texture to each colour channel, and the value within the channel defines the how much this texture should contribute to this pixel. It feels a bit like our biomemap again with a factor that defines how much a terrain generator should contribute. The main disadvantage, is there are only maximum of 4 channels in each texture, not to mention how memory inefficient it is. For a procedural terrain, I am expected to use at least 10 different sets of texture and I found it very hard to manage so many texture at once.

The idea of biomemap can be reused, a texture ID can be instead assigned to each texture. As texture comes with many different categories, like albedomap, normalmap, roughnessmap etc., these different categories are grouped together and they all share the same texture ID for easier lookup later.

To generate a texture splatmap, I use a rule-based approach. There are currently altitude and gradient rule in this system. They are pretty trivial and enable texture when the current altitude and gradient meets the condition.

### Texture definition language

Next problem is how to manage so many rules? As a user I don't want to type all the rules in the source file and whenever I modify any of them I need to recompile the whole application, this makes debugging much more time-consuming.

How about defining all the rules in an external file and parse them in runtime? Sounds like a good idea. There are ways to do this such as using comma separated value, but I want it to be more flexible and human-readable. And then I design a simple language to define these rules.

This language allows user to declare texture names, these names are just symbols and users are still required to load texture data during runtime themselves. The language comes with two rule sets, being altitude and gradient rule. Rules are assigned to biomes, and each biome can more than one rule, and there are no limit how user wants to configure it.

These rules are parsed by a simple lexer and parser, I built it from scratch because it is very fun to do so rather using a library. And finally I store all these rules using a database so I can lookup rules with SQL queries which is much simpler than using a multi-key hash table.

With all the rules a splatmap can be generated. Using the same principle as in biome generation, texture is directly assigned to the terrain based on texture ID, here is the result.

### Rule-based biome-dependent texture splatting

This is totally expected because there is no information about how much each texture should contribute to a pixel. I can simply reuse the single histogram filter developed earlier to smooth it. However sending a large histogram to the shader is much slower, because terrain generation happens only once and I can reuse the result while in the shader this is done on per-frame basis.

### Percentage-closer filtering

For texture there is usually no need to use a very large kernel radius so a simple convolution should be sufficient in runtime. Do you still remember PCF earlier, splatmap has the same principle as the shadow map and biomemap for being not applicable for linear interpolation.

One of the major drawbacks of PCF is, it creates colour banding, as shown in the screenshot.

### Stratified sampling

Fortunately there are ways to solve this problem. For shadow map there are other filtering techniques available, this particular technique improves sampling pattern, rather than taking uniform grid samples, sampling points are displaced inside the cell randomly before mapping onto a disk. So basically, colour banding effect is replaced with high-frequency noise.

*Next slide*

I decide to use value noise here rather than high-frequency noise proposed in the original paper, the results look promising and colour bands are no longer visible. Unless the camera is zoomed in, visual artefacts are negligible.

*Next slide*

Here is a very close look at the edge of texture region in case the previous screenshot is not clear, there are some whirl-like splines, but in general it is acceptable for me.

## Terrain generation pipeline

To give a short summary of all previous works, they are put together and arranged into a pipeline. User sends a request about the chunk location they want, and the chunk pipeline will check if it is available in the cache, if so return immediately, otherwise the pipeline shall generate one.

Rather than using a linear pipeline, results of each generator are returned back to the chunk provider so that all operations can be done asynchronously overlapping and the external user can choose to not wait for work to be done, and come back later, to achieve a non-blocking just-in-time terrain generation. This architecture also makes working with the free-slip system easier, smoothing the biomemap and erosion both requires knowledge about the surrounded chunks, so that one generator can release control as soon as the work is done, instead of waiting for the entire pipeline to finish.

## Sky generation

I have been working on the terrain for long time, how about shifting the focus onto something else, for instance let's look up into the sky. Right now the sky is shaded using static environment mapping which is very boring and unrealistic.

Sky is a mystery, there are a lot of science behind when we are talking about "how is the weather today?". Why the sky has to be blue not green? Why the number of hour during summer is different from that during winter? Why the horizon on the sea looks misty? What are clouds?

Because photorealistic-rendering is such a large topic, I created a separate engine dedicated for this and hopefully it can help us answering these questions later on.

### Atmospheric scattering

Thanks to scattering, our sky has a wide range of colour and it changes throughout the day. Scientists have discovered mainly two types of scattering; Rayleigh scattering happens mainly at molecular level where the radius of particle is much smaller than the wavelength of light, whereas Mie scattering involves particles much larger than the wavelength of light.

For Rayleigh scattering, light with different wavelengths have different scattering factor, and it depends on the type of molecules. For our Earth, air is mostly consist of nitrogen, which scatters high frequency waves more. For Mie scattering, when the particles become larger, it makes little difference for different wavelengths, such that lights are scattered away almost evenly.

### Sun

Now we have scattering equations, how to determine the position of the sun? I try to model how the Earth orbits around the Sun and it is in fact pretty easy to do so. The seasonal effects exist because of axial tilt, this causes one pole to be directed more toward the Sun on one side of the orbit, and the other pole on the other side. This can be modelled using a simple formula.

### Implementation

Good, these information should be enough to model a sky. Scattering is a very expensive operation because it performs simulation on molecular level, also precomputed lookup table is not preferable for procedural generation, therefore some trade-off must be made so it can be done in real-time.

One perfect solution is rendering with ray marching. It divides each ray into smaller ray segments, sampling some functions at each step. In atmospheric scattering, two-step ray marching is performed. Each segment in the primary ray generates a new secondary ray, and samples the scattering values at each secondary segment. Finally accumulate all the results. And our poor GPU needs to repeat this process for every pixel every frame.

*Next slide*

And here is how it looks. Notice the sky and sun colours are simulated quite successfully, and we got the red-orange sunrise and blue sky at noon. But there is another problem, the sharp edge makes it looks more like a bomb than a sun.

The problem here is because the simulation is physically-based, and the equation does not know anything about the intensity. The displayed intensity is clamped between zero and one while the sun intensity can go way beyond this range.

### Tone mapping

The solution is by applying a tone mapping function, it is maps any positive real number to the displayable range.

Tone mapping function can be as trivial as a simple exponential equation or a reciprocal. One of the most widely used tone mapping function is called Reinhard equation. While it is fast, it suffers from the problem of dark area desaturation so modern graphics application tends to use more advanced technique. By inspecting the sketch, you will notice when x equals 1 the outputs are just around 0.5 and most of the dynamic range are unused.

### Filmic tone mapping

Hence, artists developed new sets of tone mapping function for photorealistic-rendering, so called filmic tone mapping. These functions uses noticeably more dynamic range than Reinhard functions and usually provide a clearer, more saturated image.

In the rendering engine, I implemented a few of these functions and allow users to choose them in runtime.

### HDR

With that being said, after applying filmic tone mapping everything looks much better. I also includes gamma correction here.

*Next slide*

Things get more interesting if I am trying to use scattering parameters from other planets. Just a quick note on the dark line at the horizon, that's the edge of atmosphere rather than visual artefacts.

## Lighting

Now we have the main light source, let's add some lighting effects on the terrain. This is simply done using Blinn-Phong shading model. I have also included texture normalmap blended with terrain normal so we can see the erosion details on the terrain and texture details at the same time.

The colour of the light also changes based on colour of the sun and sky at the moment.

## Shadow mapping

With lighting, the next thing to be considered is adding shadows. The modern way of rendering shadow is by using shadow mapping. Instead of rendering from the camera, objects are rendered from light's perspective and store the distance from any pixel to light into depth buffer. This buffer can then be used to test from camera's perspective if any pixel is occluded.

### Cascaded shadow mapping

The basic idea is simple, but it gets a bit more complicated as we are handling a directional light shadow, because there is no precise position defined but a light direction.

How about assigning a position along the light direction? If the light is closer to an object more details are obtained however the range of shadow becomes very limited, and that will become the opposite if the light is further away.

If one shadow map is not enough, how about having more than one and choose based on the level of details we need?

This technique divides camera view frustum into a few sub-frustums, and finds a tight bounding box around each of them as light space frustum. When shadow is rendered, choose which frustum to be used based on the distance from the viewer.

*Next slide*

Here is how the terrain looks with shadow. The screenshot is taken in the morning and the mountain behind occludes the scene pretty nicely.

*Next slide*

Another sets of screenshot showing how the system switches between different shadow maps based on the distance from camera. The lowest level means the best quality.

### Soft shadow

# Project management