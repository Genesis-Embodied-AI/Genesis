# USD Parser Specification

This document describes the specification of the USD parser in Genesis.

# UsdArticulation Load Strategy

## About Scaling

We use $T$ to represent a transform considering Rotation $R$, Scaling $S$ and translation $t$, while $Q$ represents a transform only considering $R$ and $t$.

So a $T$ can be written as:

$$
T = Q \cdot S
$$

## Usd Stage Tree Structure

`Transform` on `Prim` is a local transform according to its parent. We use $T_i^j$ to describe it, where $j$ indicates the `Prim` and $i$ indicates the parent `Prim` of `Prim` $j$. 

To prevent nested transform, we consider $T_j^w$ as the global transform of $j$.

Thus, any relative transform can be calculated as:

$$
T^w_j = T^w_i \cdot T^i_j \\
T^i_j = ({T^w_i})^{-1} \cdot T^w_j
$$

## Genesis (Gs) Tree Structure

`Transform` on `link` is a local transform according to its parent (but no scaling). We use $Q_i^j$ to describe it, where $j$ indicates the `link` and $i$ indicates the parent `link` of `link` $j$. 

To prevent nested transform, we consider $Q_j^w$ as the global transform of $j$.

Thus, any relative transform can be calculated as:

$$
Q^w_j =Q^w_i \cdot Q^i_j \\
Q^i_j = ({Q^w_i})^{-1} \cdot Q^w_j
$$

## Between Usd and Gs?

There is no typical relationship between $T^i_j$ and $Q^i_j$; relative transforms provide no general relationship. This limitation arises from the complexity of tree structures and nested relationships.

The only relationship between $T$ and $Q$ is in the world space, which is:

$$
T^w_i = Q^w_{i'} \cdot S^{i'}_i
$$

In Gs, the $S^{i'}_{i}$ will be left to transform the `Mesh` on `link` $i'$.

## Transform to World Space

In Usd, the joint is described using $T_J^0$ and $T_J^1$, which tell the relative transform of joint $J$ w.r.t. Link $0$ and $1$. 

[https://openusd.org/dev/api/usd_physics_page_front.html#usdPhysics_jointed_bodies](https://openusd.org/dev/api/usd_physics_page_front.html#usdPhysics_jointed_bodies)

The joint axis can only be chosen from $X$, $Y$, or $Z$, specified by the string `'X'/'Y'/'Z'`. We use $\hat{e}$ to represent it.

NOTE: The axis is defined in both links' local space.

### Joint Axis

Axis in world space:

$$
\begin{bmatrix}
\hat{e}^w \\
0
\end{bmatrix} 
= T^w_0 \cdot T^0_{J} \cdot 
\begin{bmatrix}
\hat{e} \\
0
\end{bmatrix}
$$

$$
\begin{bmatrix}
\hat{e}^w \\
0
\end{bmatrix} 
= T^w_1 \cdot T^1_{J} \cdot 
\begin{bmatrix}
\hat{e} \\
0
\end{bmatrix}
$$

Convert to Genesis Link 1 Local Space (Genesis-Style). For conciseness, the homogeneous 0 is ignored.

$$
\hat{e}^{1'} = (Q^w_{1'})^{-1} \cdot \hat{e}^w
$$

### Joint Position

Position in world space:

$$
\begin{bmatrix}
P^w \\
1
\end{bmatrix} 
= T^w_0 \cdot T^0_{J} \cdot 
\begin{bmatrix}
P \\
1
\end{bmatrix}
$$

$$
\begin{bmatrix}
P^w \\
1
\end{bmatrix} 
= T^w_1 \cdot T^1_{J} \cdot 
\begin{bmatrix}
P \\
1
\end{bmatrix}
$$

Convert to Genesis Link 1 Local Space. For conciseness, the homogeneous 1 is ignored.

$$
P^{1'} = (Q^w_{1'})^{-1} \cdot P^w
$$

### Distance Limit Scaling

$$
\beta \| \hat{e}^{1'} \| = \| \hat{e}^w \| = \alpha \|\hat{e}\|
$$

Because $Q^w_{1'}$ keeps the distance (Rigid Transform), and $\|\hat{e}\|$ is $1$ by definition, we have:

$$
\beta = \alpha = \| \hat{e}^w \|
$$

The distance limit should be scaled by $\beta$.

Unfortunately, if parent and child links are not at the same scale, the distance limit is difficult to determine, and it is unclear which scale to choose.

📌 Currently, the distance limit is not scaled and is kept as-is (world space size). 

### Angle Limit

Under **homogeneous scaling**, the angle limit is preserved. We assume the **synthesis** transform is **homogeneous scaling**.

📌 Currently, the angle limit is not modified and is kept as-is (world space size).