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

### Zoom algorithm

The original library is written in Java and only works on a single thread. With license agreement I ported it to C++ and improve it a bit so that it can be executed in parallel.

So now we have a biome map, the only thing left is assigning different generators based on biome ID. It should be simple, right?

*Next slide*

Oh wow, looks like we just created some cliffs. It turns out to be not so simple.