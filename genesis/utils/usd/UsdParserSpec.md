# USD Parser Specification

This document describes the specification of the USD parser in Genesis.

## Scaling

USD allows scaling in transformation matrices, but Genesis `link` transforms are **rigid only** (rotation and translation, no scaling). This fundamental architectural difference requires decomposing USD transforms to separate scaling from the rigid component.

We decompose any USD transform $T$ (which includes rotation $R$, scaling $S$, and translation $t$) into a rigid transform $Q$ (rotation $R$ and translation $t$ only) and a scale matrix $S$:

$$
T = Q \cdot S
$$

where $Q$ is used for the Genesis `link` transform, and $S$ is **baked into the mesh geometry** to preserve the visual appearance.

## Tree Structure and Transform Notation

Both USD and Genesis use hierarchical tree structures where each node has a local transform relative to its parent.

**USD Stage:** We use $T_i^j$ to denote the transform from parent `Prim` $i$ to child `Prim` $j$ (includes rotation, scaling, and translation). The world-space transform $T_j^w$ of `Prim` $j$ is computed as:

$$
T^w_j = T^w_i \cdot T^i_j \Rightarrow T^i_j = ({T^w_i})^{-1} \cdot T^w_j
$$

**Genesis:** We use $Q_i^j$ to denote the transform from parent `link` $i$ to child `link` $j$ (rigid transform only, no scaling). The world-space transform $Q_j^w$ of `link` $j$ is computed as:

$$
Q^w_j = Q^w_i \cdot Q^i_j \Rightarrow Q^i_j = ({Q^w_i})^{-1} \cdot Q^w_j
$$

## Relationship Between USD and Genesis

**Notation:** We use $i$ to denote a USD `Prim`, and $i'$ to denote the corresponding Genesis `link` (with scale baked).

There is no direct relationship between $T^i_j$ and $Q^{i'}_{j'}$. This limitation arises from the complexity of tree structures and nested relationships. The only relationship between $T$ and $Q$ is in world space, which is:

$$
T^w_i = Q^w_{i'} \cdot S^{i'}_i
$$

where $T^w_i$ is the world-space transform of USD `Prim` $i$, $Q^w_{i'}$ is the world-space transform of the corresponding Genesis `link` $i'$, and $S^{i'}_i$ is the scale matrix that transforms from Genesis link $i'$ to USD prim $i$. In Genesis, the scale $S^{i'}_{i}$ is baked into the meshes on `link` $i'$.

## Transform to World Space

In USD, a joint is described using $T_J^{l_0}$ and $T_J^{l_1}$, which represent the relative transforms of joint $J$ with respect to Link $l_0$ (specified by `Body0Rel`) and Link $l_1$ (specified by `Body1Rel`).

Reference: [USD Physics Jointed Bodies](https://openusd.org/dev/api/usd_physics_page_front.html#usdPhysics_jointed_bodies)

The joint axis can only be chosen from the $X$, $Y$, or $Z$ axes, specified by the string `'X'`, `'Y'`, or `'Z'`. We use $\hat{e}$ to represent the axis vector.

**Note:** The axis is defined in both links' local coordinate spaces.

## Joint Transform to Genesis

For both the joint axis and position, we can use the transforms of either link to calculate the world-space value (in practice we use $l_1$), then convert to Genesis local space.

**Joint Axis:**
$$
\begin{bmatrix}
\hat{e}^w \\
0
\end{bmatrix} 
= T^w_{l_0} \cdot T^{l_0}_{J} \cdot 
\begin{bmatrix}
\hat{e} \\
0
\end{bmatrix}, \;\;\;\;
\begin{bmatrix}
\hat{e}^w \\
0
\end{bmatrix} 
= T^w_{l_1} \cdot T^{l_1}_{J} \cdot 
\begin{bmatrix}
\hat{e} \\
0
\end{bmatrix}
$$

$$
\hat{e}' = \hat{e}^{l_1'} = (Q^w_{l_1'})^{-1} \cdot \hat{e}^w
$$

**Joint Position:**
$$
\begin{bmatrix}
P^w \\
1
\end{bmatrix} 
= T^w_{l_0} \cdot T^{l_0}_{J} \cdot 
\begin{bmatrix}
P \\
1
\end{bmatrix}, \;\;\;\;
\begin{bmatrix}
P^w \\
1
\end{bmatrix} 
= T^w_{l_1} \cdot T^{l_1}_{J} \cdot 
\begin{bmatrix}
P \\
1
\end{bmatrix}
$$

$$
P' = P^{l_1'} = (Q^w_{l_1'})^{-1} \cdot P^w
$$