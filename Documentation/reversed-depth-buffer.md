# Reversed Depth Buffer

To mitigate the problem of losing floating point precision as the object gets further away from the camera, especially in a large scene like terrain rendering, it is useful to reverse the depth buffer to evenly distribute depth precision across the depth range. The rendering engine `SuperRealism+` utilises this technique by default.

This technique does come with one major drawback of increasing implementation and code review difficulties when working with depth buffer as all depth logics need to be considered carefully in a way that violates common sense. It is strongly recommended that developers should have clear knowledge in different rendering techniques.

## Advice

### Reversed depth mapping

The key take-away of reversed depth buffer is any object closer to the camera has increasing depth, rather than decreasing as in common sense. Therefore, depth buffer is cleared to zero, which means the object is infinitely far away initially, rather than one; and depth test is performed using a *greater than* logic as object closer to the camera has greater depth. This affects both depth used in regular scene rendering and shadow mapping.

In the special scenario of shadow mapping, it is common to apply a bias value to reduce shadow acne artefacts by moving a fragment towards the light. To achieve this, the fragment depth is added with instead of normally deducted by, a small bias value.

As the mapping of the depth is reversed, all use of projection matrix have the near and far plane parameters exchanged. This is done internally in the engine and use of any camera class managed by the renderer like `STPCamera` is not affected, i.e., parameters are provided as is; but one should pay attention when using a custom camera class and calculating projection matrix.

### Clip volume

Reversed depth buffer works well for depth within the range of [0.0, 1.0] while OpenGL by default uses [-1.0, 1.0]. Clip volume is thus modified to match that as in DirectX, and therefore the normalised device coordinate (NDC) is now defined as:

$$
NDC.xy \in [-1.0, 1.0]
$$

$$
NDC.z \in [0.0, 1.0]
$$