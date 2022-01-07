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

So how should we get a flat plane, and remember, the terrain needs to be infinite.

### Noise

The next problem is to generate the height field. A common way is by using a noise. In this project a perlin-simplex hybrid noise with my own improvement is used to allows better control to randomness.

### Hydraulic processing