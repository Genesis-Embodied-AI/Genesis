# USD Parser Specification

This document describes the specification of the USD parser in Genesis.

# UsdArticulation Load Strategy

# Takeaway

- **Global Transform Is The First-Class Citizen**
- In Physics, **World Cooridinates Give The Equivalence.**

## About Scaling

We use $T$ to present a transform considering Rotation $R$, Scaling $S$ and translation $t$, while use $Q$ to present a transform only considering $R$ and $t$.

So a $T$ can be written as:

$$
T = Q \cdot S
$$

## Usd Stage Tree Structure

`Transform` on `Prim` is a local transform according to it’s parent, I’d like to use $T_i^j$ to describe it. $j$ indicates the `Prim`, $i$ indicates the parent `Prim` of $j$. 

To prevent nested transform, we consider $T_j^w$ as the global transform of $j$.

Thus, any related transform can be calculated as:

$$
T^w_j = T^w_i \cdot T^i_j \\
T^i_j = ({T^w_i})^{-1} \cdot T^w_j
$$

## Genesis (Gs) Tree Structure

`Transform` on `link` is a local transform according to it’s parent (but no scaling), I’d like to use $T_i^j$ to describe it. $j$ indicates the `link`, $i$ indicates the parent `link` of $j$. 

To prevent nested transform, we consider $Q_j^w$ as the global transform of $j$.

Thus, any related transform can be calculated as:

$$
Q^w_j =Q^w_i \cdot Q^i_j \\
Q^i_j = ({Q^w_i})^{-1} \cdot Q^w_j
$$

## Between Usd & Gs?

There is no typical relationship between $T^i_j$ and $Q^i_j$, relative transform gives nothing in general. 💢 That’s why I hate Tree Structure and Nested Relationship!

The only relationship between $T$ and $Q$  is in the world space, which is:

$$
T^w_i = Q^w_{i'} \cdot S^{i'}_i
$$

In Gs, the $S^{i'}_{i}$ will be applied to `Mesh` (points).

## Transform Anything to WORLD SPACE! 👍

In Usd, the joint is described using $T_J^0$ and $T_J^1$, which tell the related transform of joint $J$ w.r.t. Link $0$ and $1$. 

[https://openusd.org/dev/api/usd_physics_page_front.html#usdPhysics_jointed_bodies](https://openusd.org/dev/api/usd_physics_page_front.html#usdPhysics_jointed_bodies)

The axis of joint can only be choosed from $X,Y,Z$, specified by string `'X'/'Y'/'Z'`. We use $\hat{e}$ to present it.

NOTE: The axis defined in both 2 links’ local space.

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

Convert to Genesis Link 1 Local Space (Genesis-Style), for concise, the homogeneous 0 is ignored.

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

Convert to Genesis Link 1 Local Space, for concise, the homogeneous 1 is ignored.

$$
P^{1'} = (Q^w_{1'})^{-1} \cdot P^w
$$

### Distance Limit Scaling

$$
\beta \| \hat{e}^{1'} \| = \| \hat{e}^w \| = \alpha \|\hat{e}\|
$$

Because $Q^w_{1'}$ keep the distance (Rigid Transform), and $\|\hat{e}\|$ is $1$ by definition, so we have:

$$
\beta = \alpha = \| \hat{e}^w \|
$$

The distance limit should be scaled by $\beta$.

Unfortunately, if parent and child link are not in the same scale, the distance limit is hard to determine, we don't know which one to choose.

📌 So now we don't scale the distance limit just keep it as is (world space size). 

### Angle Limit

Under **proportional scaling**, the angle limit is preserved, now we just assume the **synthesis** transform is **proportional scaling**.


📌 So now we don't modify the angle limit just keep it as is (world space size).