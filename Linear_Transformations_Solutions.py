import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear Transformations
        """
    )
    return


app._unparsable_cell(
    r"""
    import laguide as lag
    import numpy as np
    import scipy.linalg as sla
    %matplotlib inline
    import matplotlib.pyplot as plt
    from math import pi, sin, cos, sqrt
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Find the vector $T(V)$ where

        $$
        \begin{equation}
        V = \left[\begin{array}{r} 1 \\ 3 \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We use the definition of $T$ to find $T(V)$ and then check our answer by using the $\texttt{T}$ function that we defined.

        $$
        \begin{equation}
        T(V) = T \left(\left[ \begin{array}{rr} 1 \\ 3 \end{array}\right]\right)
        = \left[ \begin{array}{rr} 2(1) \\ 0 \\ (3) \end{array}\right]
        = \left[ \begin{array}{rr} 2 \\ 0 \\ 3 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(np):
    def T(V):
        W = np.zeros((3,1))
        W[0,0] = 2*V[0,0]
        W[2,0] = V[1,0]
        return W

    V = np.array([[1],[3]])
    print(T(V))
    return T, V


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Find the vector $U$ so that 

        $$
        \begin{equation}
        T(U) = \left[\begin{array}{r} 5 \\ 0 \\ -1 \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we start by taking $U$ to be an arbitrary vector in $\mathbb{R}^2$ with entries $u_1$ and $u_2$, then we get the following vector equation for $U$.

        $$
        \begin{equation}
        T(U) = T \left(\left[\begin{array}{r} u_1 \\ u_2 \end{array} \right]\right)
        = \left[\begin{array}{r} 2u_1 \\ 0 \\ u_2 \end{array} \right]
        = \left[\begin{array}{r} 5 \\ 0 \\ -1 \end{array} \right]
        \end{equation}
        $$

        We can see that $2u_1 = 5$ so $u_1 = 2.5$ and $u_2 = -1$. Therefore $U = \left[\begin{array}{r} 2.5 \\ -1 \end{array} \right]$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Write a Python function that implements the transformation $N:\mathbb{R}^3\to\mathbb{R}^2$, given by the following rule.  Use the function to find evidence that $N$ is **not linear**.

        $$
        \begin{equation}
        N \left(\left[\begin{array}{r} v_1 \\ v_2 \\ v_3 \end{array} \right]\right) = 
        \left[\begin{array}{c} 8v_2 \\  v_1 + v_2 + 3 \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(np):
    def N(V):
        W = np.zeros((2,1))
        W[0,0] = 8*V[1,0]
        W[1,0] = V[0,0] + V[1,0] + 3
        return W
    return (N,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One of the requirements for $N$ to be linear is that $N(V+W) = N(V) + N(W)$ for all vectors $V,W$ in $\mathbb{R}^3$. Any choice of $V$ and $W$ shows that this is not the case, however, and so $N$ is not linear.
        """
    )
    return


@app.cell
def _(N, np):
    V_1 = np.array([[1], [1], [1]])
    W = np.array([[1], [2], [3]])
    print(N(V_1 + W), '\n')
    print(N(V_1) + N(W))
    return V_1, W


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Consider the two transformations, $S$ and $R$, defined below.  Write a Python function that implements the composition $R\circ S$.  Explain why it is not possible to form the composition $S \circ R$.

        $$
        \begin{equation}
        S \left(\left[\begin{array}{r} v_1 \\ v_2 \\ v_3 \end{array} \right]\right) = 
        \left[\begin{array}{c}   v_1 + v_2 \\  3v_3 \end{array} \right]
        \end{equation}
        $$

        $$
        \begin{equation}
        R \left(\left[\begin{array}{r} v_1 \\ v_2  \end{array} \right]\right) = 
        \left[\begin{array}{rr} 3 &  0 \\ -1 & 1 \end{array}\right]
        \left[\begin{array}{c}   v_1 \\ v_2 \end{array} \right]
        \end{equation}
        $$
          
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We will first write a function for each of $S$ and $R$, and then we will write a function that implements the composition $R \circ S$ by first applying $S$ and then applying $R$ to the output of that transformation. We will confirm that it is working correctly by testing with an example vector.
        """
    )
    return


@app.cell
def _(np):
    def S(V):
        W = np.zeros((2, 1))
        W[0, 0] = V[0, 0] + V[1, 0]
        W[1, 0] = 3 * V[2, 0]
        return W

    def R(V):
        T = np.array([[3, 0], [-1, 1]])
        W = T @ V
        return W

    def R_composed_with_S(V):
        W = R(S(V))
        return W
    V_2 = np.array([[1], [4], [-2]])
    print(R_composed_with_S(V_2))
    return R, R_composed_with_S, S, V_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{equation}
        R \left(S\left(\left[\begin{array}{r} 1 \\ 4 \\ -2 \end{array} \right]\right)\right) 
        = R \left(\left[\begin{array}{r} 5 \\ -6 \end{array} \right]\right)
        = \left[\begin{array}{r} 3 & 0 \\ -1 & 1 \end{array} \right] \left[\begin{array}{r} 5 \\ -6 \end{array} \right]
        = \left[\begin{array}{r} 15 \\ -11 \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It is impossible to form the composition $S \circ R$ because this requires applying the transformation $S$ to the output of the mapping $R$, but $R$ outputs vectors in $\mathbb{R}^2$ and $S$ needs input vectors from $\mathbb{R}^3$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Create a Python function which implements the transformation: $S:\mathbb{R}^3\to\mathbb{R}^3$, given below. Use the function to provide evidence whether the transformation is **linear** or not.


        $$
        \begin{equation}
        S \left(\left[\begin{array}{r} v_1 \\ v_2 \\ v_3 \end{array} \right]\right) = 
        \left[\begin{array}{c} v_1 + v_2 \\  1 \\ v_3+v_1 \end{array} \right]
        \end{equation}
        $$

        Repeat for the transformation $T:\mathbb{R}^3\to\mathbb{R}^3$ is now defined by

        $$
        \begin{equation}
        T \left(\left[\begin{array}{r} v_1 \\ v_2 \\ v_3 \end{array} \right]\right) = 
        \left[\begin{array}{c} v_1 + v_2 \\  0 \\ v_3+v_1 \end{array} \right].
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**
        """
    )
    return


@app.cell
def _(np):
    def S_1(V):
        W = np.zeros((3, 1))
        W[0, 0] = V[0, 0] + V[1, 0]
        W[1, 0] = 1
        W[2, 0] = V[1, 0] + V[2, 0]
        return W
    V_3 = np.array([[1], [2], [3]])
    W_1 = np.array([[2], [1], [3]])
    print('V: \n', V_3, '\n')
    print('S(V): \n', S_1(V_3), '\n')
    print('S(W): \n', S_1(W_1), '\n')
    print('S(V+W): \n', S_1(V_3 + W_1), '\n')
    print('S(V)+S(W): \n', S_1(V_3) + S_1(W_1), '\n')
    return S_1, V_3, W_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that $S(V+W)\neq S(V) + S(W)$. Therefore, the transformation $S:\mathbb{R}^3\to\mathbb{R}^3$ is not linear.

        Now, let us consider that the transformation $T:\mathbb{R}^3\to\mathbb{R}^3$ is defined as: 
        $$
        \begin{equation}
        T \left(\left[\begin{array}{r} v_1 \\ v_2 \\ v_3 \end{array} \right]\right) = 
        \left[\begin{array}{c} v_1 + v_2 \\  0 \\ v_3+v_1 \end{array} \right]
        \end{equation}
        $$

        It only differs from the previous transformation in terms of the second entry which is zero in this case rather than 1. Let us define the corresponding python function in the code cell below:

        """
    )
    return


@app.cell
def _(np):
    def T_1(V):
        U = np.zeros((3, 1))
        U[0, 0] = V[0, 0] + V[1, 0]
        U[1, 0] = 0
        U[2, 0] = V[1, 0] + V[2, 0]
        return U
    V_4 = np.array([[1], [2], [3]])
    W_2 = np.array([[2], [4], [1]])
    print('V: \n', V_4, '\n')
    print('T(V): \n', T_1(V_4), '\n')
    print('T(W): \n', T_1(W_2), '\n')
    print('T(V+W): \n', T_1(V_4 + W_2), '\n')
    print('T(V)+T(W): \n', T_1(V_4) + T_1(W_2), '\n')
    k = 5
    print('T(kV)): \n', T_1(k * V_4), '\n')
    print('kT(V)): \n', k * T_1(V_4), '\n')
    return T_1, V_4, W_2, k


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Our conclusions differ from that in the previous case. In this case, $T(V+W) = T(V) + T(W)$ and $T(kV) = kT(V)$. Therefore, we have evidence that $T:\mathbb{R}^3\to\mathbb{R}^3$ is a linear transformation.

        Note the difference between $S$ and $T$ in this exercise.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** $L:\mathbb{R}^3\to\mathbb{R}^2$ is a **Linear Transformation** . Find $L(kU+V)$ given that $k=7$, 

        $$
        \begin{equation}
        L(U) = \left[\begin{array}{r} 1 \\ 1  \end{array} \right]\hspace{1cm}
        L(V) = \left[\begin{array}{r} 3 \\ 1  \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Since $L:\mathbb{R}^3\to\mathbb{R}^2$ is a linear transformation, we can say that $L(kU+V) = L(kU) + L(V)$. Then, we can write $L(kV) = kL(V)$. Therefore, $L(kU+V) = kL(U)+ L(V)$.

        Given $k$, $L(V)$ and $L(U)$, we can find $L(kU+V)$.

        $$
        \begin{equation}
        L(kU+V) = (7)\left[\begin{array}{r} 1 \\ 1  \end{array} \right] + 
        \left[\begin{array}{r} 3 \\ 1  \end{array} \right] = 
        \left[\begin{array}{r} 10 \\ 8  \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Let $T:\mathbb{R}^3 \to \mathbb{R}^2$ be defined by $T(X)= AX$, where 

        $$
        \begin{equation}
        A = \left[\begin{array}{rrr} 1 & 0 & 2 \\ 2 & 1 & 1  \end{array}\right].
        \end{equation}
        $$

        Find all vectors $X$ that satisfy $T(X) = \left[\begin{array}{r} 1 \\ 2  \end{array} \right]
        $. 

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Given $T(X)$ and $A$, we can find $X$ by solving the following system:

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 1 & 0 & 2 \\ 2 & 1 & 1\end{array}\right]
        \left[ \begin{array}{r} x_1 \\ x_2 \\ x_3  \end{array}\right]=
        \left[ \begin{array}{r}1 \\ 2  \end{array}\right]= T(X)
        \end{equation}
        $$

        We can see that there is no pivot in the third column of the coefficient matrix $A$, this means that $x_3$ is a free variable. Therefore, the above system has infinitely many solutions. In other words, there are an infinite number of vectors $X$ in $\mathbb{R}^3$ which get mapped to the vector $AX$ in space $\mathbb{R}^2$.



        For instance, when $x_3 = 3$, then we get $x_1 = -5$ and $x_2 = 9$.
        So, $$
        \begin{equation}
        X = \left[ \begin{array}{rrr} -5 \\ 9 \\ 3 \end{array}\right]
        \end{equation}
        $$

        """
    )
    return


@app.cell
def _(np):
    ## verifying that T(X) = AX:

    A = np.array([[1,0,2],[2,1,1]])
    X = np.array([[-5],[9],[3]])

    print("AX: \n", A@X, '\n')

    # Verifying that X is not unique:
    # When x_3 = 0, x_2 = 0, x_1 = 1

    X_2 = np.array([[1],[0],[0]])
    print("AX_2: \n", A@X_2, '\n')
    return A, X, X_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The computation demonstrates that $T(X)$ is same for two different vectors $X$. Therefore, $X$ is not unique.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Let $M:\mathbb{P}_1 \to \mathbb{P}_3$ be a transformation defined by $M(p(x)) = x^3 + p(x)$. Determine whether $M$ is linear or not. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        In order to determine whether $M$ is linear or not, let us consider two polynomials $p(x)$ and $q(x)$ in $\mathbb{P}_1$. 

        $M((p + q)(x)) = x^3 + p(x) + q(x)$ by the general properties of polynomials.

        $M(p(x)) + M(q(x)) = (x^3 + p(x)) + (x^3 + q(x)) = 2x^3 + p(x) + q(x)$.

        We can see that $M((p + q)(x))$ is not the same as $M(p(x)) + M(q(x))$. Therefore, $M$ is not linear.

        We can also see what $M(kp(x))$ and $k(M(p(x)))$ look like.

        $M(kp(x)) = x^3 + kp(x)$

        $k(M(p(x))) = kx^3 + kp(x)$

        It is clear that $M(kp(x))$ and $k(M(p(x)))$ are also different.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 9:** Let $S:\mathbb{P}_2 \to \mathbb{P}_3$ and $T:\mathbb{P}_3 \to \mathbb{P}_5$ be two **linear transformations** defined by the rules given below. Define the composition $T\circ S$ and determine whether it is linear or not. Explain why $S\circ T$ is not defined.

        $S(p(x)) = x(p(x))$ 

        $T(q(x)) = x^2(q(x))$ 

        where $p(x)$ is a polynomial in $\mathbb{P}_2$ and $q(x)$ is a polynomial in $\mathbb{P}_3$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Since $S:\mathbb{P}_2 \to \mathbb{P}_3$ and $T:\mathbb{P}_3 \to \mathbb{P}_5$ are two **linear transformations**, $T\circ S:\mathbb{P}_2 \to \mathbb{P}_5$ and it is defined by the following rule:

        $T\circ S = T(S(p(x))) = x^2(S(p(x)))$ 

        $S(p(x)) = x(p(x))$. Therefore, $T\circ S = T(S(p(x))) = x^2(xp(x))$ 

        $T\circ S = T(S(p(x))) = x^3(p(x))$

        So, $T\circ S:\mathbb{P}_2 \to \mathbb{P}_5$ is a transformation defined by the following rule:

        $T\circ S = T(S(p(x))) = x^3(p(x))$

        Let us see if  $T\circ S:\mathbb{P}_2 \to \mathbb{P}_5$ is a linear transformation.
        Consider two polynomials $p(x)$ and $q(x)$ in $\mathbb{P}_2$.

        $T(S(p(x) + q(x))) = x^3(p(x) + q(x)) = x^3(p(x)) + x^3(q(x)) $ by the properties of polynomials.

        $T(S(p(x))) + T(S(q(x))) = x^3(p(x)) + x^3(q(x)) $.

        $T(S(kp(x))) = x^3(kp(x)) = kx^3(p(x)))$ by the properties of polynomials.

        $kT(S(p(x))) = x^3(kp(x)) = kx^3(p(x)))$

        $T\circ S $ is a linear transformation since $T(S(p(x) + q(x))) =T(S(p(x))) + T(S(q(x)))$ and $T(S(kp(x))) = kT(S(p(x)))$.



        $S\circ T$ is not defined because the output space of $T$ is not the input space of $S$. The output space of $T$ is $\mathbb{P}_5$ but the input space of $S$ is $\mathbb{P}_2$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Matrix Representations
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** For each of the following linear transformations, find the standard matrix representation, and then determine if the transformation is onto, one-to-one, or invertible.

        $(a)$

        $$
        \begin{equation}
        B \left(\left[\begin{array}{r} x_1 \\ x_2 \\ x_3 \\ x_4 \end{array} \right]\right) = 
        \left[\begin{array}{c} x_1 + 2x_2 - x_3 -x_4 \\ x_2 -3x_3 +2x_4 \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $B$ maps $\mathbb{R}^4$ to $\mathbb{R}^2$ so to find the standard matrix representation of $B$, we have to first apply the formula to each of the four standard basis vectors of $\mathbb{R}^4$.

        $$
        \begin{equation}
        B\left(\left[\begin{array}{r} 1\\0\\0\\0 \end{array}\right]\right)= \left[\begin{array}{r} 1\\0 \end{array}\right] \hspace{1cm} 
        B\left(\left[\begin{array}{r} 0\\1\\0\\0 \end{array}\right]\right)= \left[\begin{array}{r} 2\\1 \end{array}\right] \hspace{1cm} 
        B\left(\left[\begin{array}{r} 0\\0\\1\\0 \end{array}\right]\right)= \left[\begin{array}{r} -1\\-3 \end{array}\right] \hspace{1cm}
        B\left(\left[\begin{array}{r} 0\\0\\0\\1 \end{array}\right]\right)= \left[\begin{array}{r} -1\\2 \end{array}\right] 
        \end{equation}
        $$

        We build $[B]$ by using these images as the columns and then find the RREF using $\texttt{FullRowReduction}$.

        $$
        \begin{equation}
        [B] = \left[\begin{array}{rr} 1 & 2 & -1 & -1 \\ 0 & 1 & -3 & 2 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    B = np.array([[1,2,-1,-1],[0,1,-3,2]])
    print(lag.FullRowReduction(B))
    return (B,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The RREF of $[B]$ has a pivot in every row, but not in the third and fourth columns. Therefore $B$ is onto but not one-to-one.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$

        $$
        \begin{equation}
        C \left(\left[\begin{array}{r} x_1 \\ x_2 \\ x_3 \end{array} \right]\right) = 
        \left[\begin{array}{c} x_1 -x_2 + 8x_3 \\ 4x_1 + 5x_2 - x_3 \\ -x_1 -x_2 + 3x_3 \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $C$ maps $\mathbb{R}^3$ to $\mathbb{R}^3$ so to find the standard matrix representation of $C$, we have to first apply the formula to each of the three standard basis vectors of $\mathbb{R}^3$.

        $$
        \begin{equation}
        C\left(\left[\begin{array}{r} 1\\0\\0 \end{array}\right]\right)= \left[\begin{array}{r} 1\\4\\-1 \end{array}\right]\hspace{1cm} 
        C\left(\left[\begin{array}{r} 0\\1\\0 \end{array}\right]\right)= \left[\begin{array}{r} -1\\5\\-1 \end{array}\right]\hspace{1cm} 
        C\left(\left[\begin{array}{r} 0\\0\\1 \end{array}\right]\right)= \left[\begin{array}{r} 8\\-1\\3 \end{array}\right]
        \end{equation}
        $$

        We build $[C]$ by using these images as the columns and then find the RREF using $\texttt{FullRowReduction}$.

        $$
        \begin{equation}
        [C] = \left[\begin{array}{rr} 1 & -1 & 8 \\ 4 & 5 & -1 \\ -1 & -1 & 3\end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    C = np.array([[1,-1,8],[4,5,-1],[-1,-1,3]])
    print(lag.FullRowReduction(C))
    return (C,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The RREF of $[C]$ has a pivot in every row and column. Therefore $C$ is onto and one-to-one, which means it is invertible.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Let $L:\mathbb{R}^3 \to \mathbb{R}^2$ be the **linear transformation** defined by $L(X)= AX$. 

        $$
        \begin{equation}
        A = \left[\begin{array}{rrr} 1 & 1 & 1\\ 2 & 3 & 4  \end{array}\right]\
        \end{equation}
        $$

        Determine whether $L:\mathbb{R}^3 \to \mathbb{R}^2$ is an invertible transformation or not.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The **linear transformation** $L:\mathbb{R}^3 \to \mathbb{R}^2$ will be invertible if the matrix $A$ is invertible as $L(X)= AX$. Since $A$ is a $ 2 \times 3 $ matrix, it cannot be invertible. Therefore, $L:\mathbb{R}^3 \to \mathbb{R}^2$ is not an invertible transformation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** $L:\mathbb{R}^3\to\mathbb{R}^2$ is a **Linear Transformation**. Find $L(X)$ given the following vectors.


        $$
        \begin{equation}
        L\left(\left[\begin{array}{r} 1\\0\\0 \end{array}\right]\right)= \left[\begin{array}{r} 2\\0 \end{array}\right] \hspace{1cm}  
        L\left(\left[\begin{array}{r} 0\\1\\0 \end{array}\right]\right)= \left[\begin{array}{r} 1\\3 \end{array}\right] \hspace{1cm}  
        L\left(\left[\begin{array}{r} 0\\0\\1 \end{array}\right]\right)= \left[\begin{array}{r} 1\\2 \end{array}\right] \hspace{1cm}
        X = \left[\begin{array}{r} 4\\5\\3 \end{array}\right]
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The standard matrix representation of $L$ is built using the images of bases vectors as columns.


        $$
        \begin{equation}
        \left[L\right] =\left[\begin{array}{rr} 2 & 1 & 1 \\ 0 & 3 & 2 \end{array}\right]  
        \end{equation}
        $$

        Now $L(X) = [L]X$.


        $$
        \begin{equation}
        L(X) = \left[L\right]X =\left[\begin{array}{rr} 2 & 1 & 1 \\ 0 & 3 & 2 \end{array}\right] \left[\begin{array}{r} 4 \\ 5 \\3 \end{array}\right] = 
        \left[\begin{array}{r} 16\\ 21\end{array}\right]
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** The standard matrix representation of a **linear transformation** $S$ is given below. Determine the input and output space of $S$ by looking at the dimensions of $\left[S\right]$. Determine whether $S$ is an invertible transformation.


        $$
        \begin{equation}
        \left[S\right] =\left[\begin{array}{rr} 2 & 0 & 3 & 8\\ 0 & 1 & 9 & 4 \end{array}\right]  
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The linear transformation is defined by $S(X)=[S]X$.  Since $[S]$ is a $2 \times 4 $ matrix, the matrix-vector product $S(X)=[S]X$ is defined only if $X$ is in $\mathbb{R}^4$.  The input space is thus $\mathbb{R}^4$.  Since the product $[S]X$ is in $\mathbb{R}^2$, the output space is $\mathbb{R}^2$.

        The transformation $S:\mathbb{R}^4\to\mathbb{R}^2$ will be invertible only if $[S]$ is an invertible matrix. Since $[S]$ is a $ 2 \times 4$ matrix it cannot be invertible, which means that $S$ is not an invertible linear transformation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** The **linear transformation** $W:\mathbb{R}^3\to\mathbb{R}^3$ is an invertible transformation. Find $X$.


        $$
        \begin{equation}
        \left[W\right] =\left[\begin{array}{rr} 1 & 1 & 0\\ 1 & 2 & 2 \\ 2 & 1 & 3 \end{array}\right] \hspace{1cm}
        W(X) = \left[\begin{array}{r} 3 \\ 11 \\ 13 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        We know that $W(X) = [W]X$, so we must solve a linear system to determin $X$.
        """
    )
    return


@app.cell
def _(lag, np):
    W_3 = np.array([[1, 1, 0], [1, 2, 2], [2, 1, 3]])
    W_X = np.array([[3], [11], [13]])
    X_1 = lag.SolveSystem(W_3, W_X)
    print('X: \n', X_1)
    return W_3, W_X, X_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Let $T:\mathbb{R}^3\to\mathbb{R}^3$ be a **linear transformation**. Given that there are two vectors in $\mathbb{R}^3$ which get mapped to the same vector in $\mathbb{R}^3$, what can you say about the number of solutions for  $[T]X=B$ ?  Explain why $T$ is not an invertible transformation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        If there are two vectors which get mapped by $T$ to the same vector in $\mathbb{R}^3$, it means that $T$ is not not one-to-one. This implies that the standard matrix representation $[T]$ contains at least one column which does not have a pivot.  This implies that there is a free variable in the system $[T]X = B$, which implies that the system has infinitely many solutions.

        Since $T$ is not one-to-one, it cannot be invertible.  If $T(U) = B$ and $T(W)=B$, then $T^{-1}(B)$ is not well-defined.  In other words, there is not a unique solution to the system $[T]X=B$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Transformations in a Plane
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** 

        ($a$) Find a matrix that represents the reflection about the $x_1$-axis. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        When we reflect anything about $x_1$-axis, then its $x_1$-coordinate remains the same while the sign of $x_2$-coordinate gets reversed.

        Let us consider that the transformation $T:\mathbb{R}^2\to\mathbb{R}^2$ represents the reflection about the $x_1$-axis. Then, the images of the basis vectors look like:


        $$
        \begin{equation}
        T\left(\left[ \begin{array}{r} 1 \\ 0  \end{array}\right]\right)= \left[ \begin{array}{r} 1 \\ 0  \end{array}\right] \hspace{1in}  T\left(\left[ \begin{array}{r} 0 \\ 1  \end{array}\right]\right)= \left[ \begin{array}{r} 0 \\ -1  \end{array}\right]  
        \end{equation}
        $$


        Therefore, the standard matrix representation of $T$ which is defined by the matrix $B$ is as follows:


        $$
        \begin{equation}
        B = \left[ \begin{array}{cc} 1 & 0 \\ 0 & -1 \end{array}\right]
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($b$) Multiply the matrix by $\texttt{coords}$ and plot the results.
        """
    )
    return


@app.cell
def _(np, plt):
    coords = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
    coords = coords.transpose()
    coords
    x = coords[0, :]
    y = coords[1, :]
    B_1 = np.array([[1, 0], [0, -1]])
    B_coords = B_1 @ coords
    x_LT2 = B_coords[0, :]
    y_LT2 = B_coords[1, :]
    fig, ax = plt.subplots()
    ax.plot(x, y, 'ro')
    ax.plot(x_LT2, y_LT2, 'bo')
    ax.plot(x, y, 'r', ls='--')
    ax.plot(x_LT2, y_LT2, 'b')
    ax.axvline(x=0, color='k', ls=':')
    ax.axhline(y=0, color='k', ls=':')
    ax.grid(True)
    ax.axis([-2, 2, -2, 2])
    ax.set_aspect('equal')
    ax.set_title('Reflection about the x1-axis')
    return B_1, B_coords, ax, coords, fig, x, x_LT2, y, y_LT2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** 

        ($a$) Find a matrix that represents the reflection about the line $x_1=x_2$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        When we reflect anything about the line $x_1 = x_2$, then its $x_1$-coordinate becomes the $x_2$-coordinate and the $x_2$-coordinate becomes the $x_1$-coordinate. This basically means that the $x_1$ and $x_2$-coordinates get exchanged.

        Let us consider that the transformation $S:\mathbb{R}^2\to\mathbb{R}^2$ represents the reflection about the line $x_1=x_2$. Then, the images of the basis vectors look like:


        $$
        \begin{equation}
        S\left(\left[ \begin{array}{r} 1 \\ 0  \end{array}\right]\right)= \left[ \begin{array}{r} 0 \\ 1  \end{array}\right] \hspace{1in}  S\left(\left[ \begin{array}{r} 0 \\ 1  \end{array}\right]\right)= \left[ \begin{array}{r} 1 \\ 0  \end{array}\right]  
        \end{equation}
        $$


        Therefore, the standard matrix representation of $S$ which is defined by the matrix $C$ is as follows:


        $$
        \begin{equation}
        C = \left[ \begin{array}{cc} 0 & 1 \\ 1 & 0 \end{array}\right]
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($b$) Multiply the matrix by $\texttt{coords}$ and plot the results.
        """
    )
    return


@app.cell
def _(np, plt):
    coords_1 = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
    coords_1 = coords_1.transpose()
    coords_1
    x_1 = coords_1[0, :]
    y_1 = coords_1[1, :]
    C_1 = np.array([[0, 1], [1, 0]])
    C_coords = C_1 @ coords_1
    x_LT2_1 = C_coords[0, :]
    y_LT2_1 = C_coords[1, :]
    fig_1, ax_1 = plt.subplots()
    ax_1.plot(x_1, y_1, 'ro')
    ax_1.plot(x_LT2_1, y_LT2_1, 'bo')
    ax_1.plot(x_1, y_1, 'r', ls='--')
    ax_1.plot(x_LT2_1, y_LT2_1, 'b')
    x_1 = np.linspace(-2, 2, 100)
    y_1 = x_1
    ax_1.plot(x_1, y_1, 'k', ls='dashed')
    ax_1.axvline(x=0, color='k', ls=':')
    ax_1.axhline(y=0, color='k', ls=':')
    ax_1.grid(True)
    ax_1.axis([-2, 2, -2, 2])
    ax_1.set_aspect('equal')
    ax_1.set_title('Reflection about the line x_1 = x_2')
    return C_1, C_coords, ax_1, coords_1, fig_1, x_1, x_LT2_1, y_1, y_LT2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** 

        ($a$) Find a matrix that represents the rotation clockwise by an angle $\theta$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The matrix which represents the rotation clockwise by an angle $\theta$ is as follows:

        $$
        \begin{equation}
        M = \left[ \begin{array}{cc} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($b$) Let $\theta = 90^{\circ}$.  Multiply the matrix by $\texttt{coords}$ and plot the results.
        """
    )
    return


@app.cell
def _(cos, np, pi, plt, sin):
    coords_2 = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
    coords_2 = coords_2.transpose()
    coords_2
    x_2 = coords_2[0, :]
    y_2 = coords_2[1, :]
    theta = pi / 2
    M = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    M_coords = M @ coords_2
    x_LT3 = M_coords[0, :]
    y_LT3 = M_coords[1, :]
    fig_2, ax_2 = plt.subplots()
    ax_2.plot(x_2, y_2, 'ro')
    ax_2.plot(x_LT3, y_LT3, 'bo')
    ax_2.plot(x_2, y_2, 'r', ls='--')
    ax_2.plot(x_LT3, y_LT3, 'b')
    ax_2.axvline(x=0, color='k', ls=':')
    ax_2.axhline(y=0, color='k', ls=':')
    ax_2.grid(True)
    ax_2.axis([-2, 2, -2, 2])
    ax_2.set_aspect('equal')
    ax_2.set_title('Clockwise Rotation')
    return M, M_coords, ax_2, coords_2, fig_2, theta, x_2, x_LT3, y_2, y_LT3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** 

        $(a)$ Find a matrix that represents a vertical shear followed by the rotation in Example 3. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we take $k = 3$ as our vertical shearing factor then the matrix that represents this vertical shear followed by the rotation in Example 3 is the following

        $$
        \begin{equation}
        \left[\begin{array}{rr} \cos(\frac{\pi}{6}) & -\sin(\frac{\pi}{6}) \\ \sin(\frac{\pi}{6}) & \cos(\frac{\pi}{6}) \end{array}\right] \left[\begin{array}{rr} 1 & 0 \\ 3 & 1\end{array}\right]
        = \left[\begin{array}{rr} \cos(\frac{\pi}{6}) - 3\sin(\frac{\pi}{6}) & -\sin(\frac{\pi}{6}) \\ \sin(\frac{\pi}{6}) + 3\cos(\frac{\pi}{6}) & \cos(\frac{\pi}{6}) \end{array}\right]
        = \left[\begin{array}{rr} \frac{-3 + \sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1 + 3\sqrt{3}}{2} & \frac{\sqrt{3}}{2}\end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(cos, np, pi, sin):
    R_1 = np.array([[cos(pi / 6), -sin(pi / 6)], [sin(pi / 6), cos(pi / 6)]])
    S_2 = np.array([[1, 0], [3, 1]])
    RS = R_1 @ S_2
    print(RS)
    return RS, R_1, S_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$ Multiply the matrix by $\texttt{coords}$ and plot the results.
        """
    )
    return


@app.cell
def _(RS, np, plt):
    coords_3 = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
    coords_3 = coords_3.transpose()
    x_3 = coords_3[0, :]
    y_3 = coords_3[1, :]
    RS_coords = RS @ coords_3
    x_LT1 = RS_coords[0, :]
    y_LT1 = RS_coords[1, :]
    fig_3, ax_3 = plt.subplots()
    ax_3.plot(x_3, y_3, 'ro')
    ax_3.plot(x_LT1, y_LT1, 'bo')
    ax_3.plot(x_3, y_3, 'r', ls='--')
    ax_3.plot(x_LT1, y_LT1, 'b')
    ax_3.axvline(x=0, color='k', ls=':')
    ax_3.axhline(y=0, color='k', ls=':')
    ax_3.grid(True)
    ax_3.axis([-2, 2, -1, 3])
    ax_3.set_aspect('equal')
    return RS_coords, ax_3, coords_3, fig_3, x_3, x_LT1, y_3, y_LT1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Create a new matrix of coordinates and apply one of the transformations in the Examples.  Plot the results.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As an example, we use the following matrix of coordinates and the matrix $A$ that causes a horizontal stretch.
        """
    )
    return


@app.cell
def _(np, plt):
    coords_4 = np.array([[-1, 0], [0, 1], [1, 0], [1, -1], [-2, -2], [-1, 0]])
    coords_4 = coords_4.transpose()
    x_4 = coords_4[0, :]
    y_4 = coords_4[1, :]
    A_1 = np.array([[2, 0], [0, 1]])
    A_coords = A_1 @ coords_4
    x_LT1_1 = A_coords[0, :]
    y_LT1_1 = A_coords[1, :]
    fig_4, ax_4 = plt.subplots()
    ax_4.plot(x_4, y_4, 'ro')
    ax_4.plot(x_LT1_1, y_LT1_1, 'bo')
    ax_4.plot(x_4, y_4, 'r', ls='--')
    ax_4.plot(x_LT1_1, y_LT1_1, 'b')
    ax_4.axvline(x=0, color='k', ls=':')
    ax_4.axhline(y=0, color='k', ls=':')
    ax_4.grid(True)
    ax_4.axis([-5, 3, -3, 3])
    ax_4.set_aspect('equal')
    return A_1, A_coords, ax_4, coords_4, fig_4, x_4, x_LT1_1, y_4, y_LT1_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** 

        ($a$) Construct a matrix that represents a horizontal and a vertical stretch by a factor of 2 . 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Let us consider that the transformation $L:\mathbb{R}^2\to\mathbb{R}^2$ represents a horizontal and a vertical stretch by a factor of 2 . Then, the images of the basis vectors look like:


        $$
        \begin{equation}
        L\left(\left[ \begin{array}{r} 1 \\ 0  \end{array}\right]\right)= \left[ \begin{array}{r} 2 \\ 0  \end{array}\right] \hspace{1in}  L\left(\left[ \begin{array}{r} 0 \\ 1  \end{array}\right]\right)= \left[ \begin{array}{r} 0 \\ 2
        \end{array}\right]  
        \end{equation}
        $$

        Therefore, the standard matrix representation of $L$ which is defined by the matrix $D$ is as follows:


        $$
        \begin{equation}
        D = \left[ \begin{array}{cc} 2 & 0 \\ 0 & 2 \end{array}\right]
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($b$) Create a new matrix of coordinates.  Apply this transformation and plot the results.
        """
    )
    return


@app.cell
def _(np, plt):
    coord = np.array([[0, 0], [0, 2], [3, 4], [2, 0], [0, 0]])
    coord = coord.transpose()
    x_5 = coord[0, :]
    y_5 = coord[1, :]
    D = np.array([[2, 0], [0, 2]])
    D_coord = D @ coord
    x_LT1_2 = D_coord[0, :]
    y_LT1_2 = D_coord[1, :]
    fig_5, ax_5 = plt.subplots()
    ax_5.plot(x_5, y_5, 'ro')
    ax_5.plot(x_LT1_2, y_LT1_2, 'bo')
    ax_5.plot(x_5, y_5, 'r', ls='--')
    ax_5.plot(x_LT1_2, y_LT1_2, 'b')
    ax_5.axvline(x=0, color='k', ls=':')
    ax_5.axhline(y=0, color='k', ls=':')
    ax_5.grid(True)
    ax_5.axis([-5, 10, -5, 10])
    ax_5.set_aspect('equal')
    ax_5.set_title('Horizontal and Vertical Stretch')
    return D, D_coord, ax_5, coord, fig_5, x_5, x_LT1_2, y_5, y_LT1_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Applications

        #### Computer Graphics
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** 

        $(a)$ Find a single matrix that represents a transformation that has the effect of a reflection about the line $x_1=x_2$ followed by a shift four units to the left. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We will find matrices that do each part seperately then compose them to get a single matrix that does everything. Since a translation four units to the left cannot be expressed using a $2 \times 2$ matrix, we will need to use the same trick as before and view $\mathbb{R}^2$ as a plane embedded in $\mathbb{R}^3$. If we associate the coordinates $(x_1,x_2)$ of a point in $\mathbb{R}^2$ with the coordinates $(x_1,x_2,1)$ of a point in $\mathbb{R}^3$ and define the matrix $L$ below, then multiplying any vector in the plane $x_3 = 1$ with $L$ shears the vector parallel to the $x_1x_2$-plane, in the direction of the desired translation.

        $$
        \begin{equation}
        L = \left[\begin{array}{rr} 1 & 0 & -4 \\ 0 & 1 & 0 \\ 0 & 0 & 1  \end{array}\right]
        \end{equation}
        $$

        Reflecting about the line $x_1 = x_2$ is equivalent to first rotating by $\frac{\pi}{4}$ (since the line $x_1 = x_2$ is $\frac{\pi}{4}$ off from the vertical axis), reflecting over the vertical axis, then rotating back by $-\frac{\pi}{4}$. This can be represented by the following composition.

        $$
        \begin{equation}
        \left[\begin{array}{rr} \cos(\frac{\pi}{4}) & -\sin(\frac{\pi}{4}) \\ \sin(\frac{\pi}{4}) & \cos(\frac{\pi}{4}) \end{array}\right] 
        \left[\begin{array}{rr} 1 & 0 \\ 0 & -1\end{array}\right]
        \left[\begin{array}{rr} \cos(-\frac{\pi}{4}) & -\sin(-\frac{\pi}{4}) \\ \sin(-\frac{\pi}{4}) & \cos(-\frac{\pi}{4}) \end{array}\right]
        = \left[\begin{array}{rr} 0 & 1 \\ 1 & 0\end{array}\right]
        \end{equation}
        $$

        Since we want to compose these transformations, we need to find a way of interpreting the reflection as acting on vectors in the plane $\mathbb{R}^2$ embedded in $\mathbb{R}^3$. We can do this in exactly the same way as above and we get that the appropriate matrix that represents a reflextion about the line $x_1 = x_2$ in the plane $x_3 = 1$ is the following matrix $R$.

        $$
        \begin{equation}
        R = \left[\begin{array}{rr} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{array}\right]
        \end{equation}
        $$

        Composing these two to get our final matrix gives us the following matrix.

        $$
        \begin{equation}
        LR = \left[\begin{array}{rr} 1 & 0 & -4 \\ 0 & 1 & 0 \\ 0 & 0 & 1  \end{array}\right]
        \left[\begin{array}{rr} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{array}\right]
        = \left[\begin{array}{rr} 0 & 1 & -4 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$ Apply the transformation to the shape defined by $\texttt{coords}$ and plot the results.
        """
    )
    return


@app.cell
def _(np, plt):
    coords_5 = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
    coords_5 = coords_5.transpose()
    x_6 = coords_5[0, :]
    y_6 = coords_5[1, :]
    Ones = np.ones((1, 5))
    coords_5 = np.vstack((x_6, y_6, Ones))
    LR = np.array([[0, 1, -4], [1, 0, 0], [0, 0, 1]])
    coords_transformed = LR @ coords_5
    x_transformed = coords_transformed[0, :]
    y_transformed = coords_transformed[1, :]
    fig_6, ax_6 = plt.subplots()
    ax_6.plot(x_6, y_6, 'ro')
    ax_6.plot(x_transformed, y_transformed, 'bo')
    ax_6.plot(x_6, y_6, 'r', ls='--')
    ax_6.plot(x_transformed, y_transformed, 'b')
    ax_6.axvline(x=0, color='k', ls=':')
    ax_6.axhline(y=0, color='k', ls=':')
    ax_6.grid(True)
    ax_6.axis([-5, 2, -1, 2])
    ax_6.set_aspect('equal')
    ax_6.set_title('Reflection then Translation')
    return (
        LR,
        Ones,
        ax_6,
        coords_5,
        coords_transformed,
        fig_6,
        x_6,
        x_transformed,
        y_6,
        y_transformed,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:**  

        $(a)$ Find a single matrix that represents a rotation about the point $(1,2)$.  (*Hint:  Make use of a translation to bring the center of rotation to $(0,0)$*.)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Rotating about the point $(1,2)$ is equivalent to translating one unit to the left and two units down, performing the desired rotation, and then translating back one unit to the right and two units up. Since a translation cannot be expressed using a $2 \times 2$ matrix, we will need to use the same trick as before and view $\mathbb{R}^2$ as a plane embedded in $\mathbb{R}^3$. If we associate the coordinates $(x_1,x_2)$ of a point in $\mathbb{R}^2$ with the coordinates $(x_1,x_2,1)$ of a point in $\mathbb{R}^3$ and define the matrices $L$ and $R$ below, then multiplying any vector in the plane $x_3 = 1$ with $L$ translates it one unit left and two units down, while multiplying by $R$ translates it back.

        $$
        \begin{equation}
        L = \left[\begin{array}{rr} 1 & 0 & -1 \\ 0 & 1 & -2 \\ 0 & 0 & 1  \end{array}\right] \hspace{1cm}
        R = \left[\begin{array}{rr} 1 & 0 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 1  \end{array}\right]
        \end{equation}
        $$

        We know that to rotate a vector by an angle $\theta$ we must multiply by the matrix below.

        $$
        \begin{equation}
        \left[ \begin{array}{cc} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{array}\right]
        \end{equation}
        $$

        Since we want to compose these transformations, we need to find a way of interpreting the rotation as acting on vectors in the plane $\mathbb{R}^2$ embedded in $\mathbb{R}^3$. We can do this in exactly the same way as above and we get that the appropriate matrix that represents a rotation in the plane $x_3 = 1$ is the following matrix $S$.

        $$
        \begin{equation}
        S = \left[ \begin{array}{cc} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{array}\right]
        \end{equation}
        $$

        Composing these three to get our final matrix gives us the following matrix.

        $$
        \begin{equation}
        RSL = \left[\begin{array}{rr} 1 & 0 & 1 \\ 0 & 1 & 2 \\ 0 & 0 & 1  \end{array}\right]
        \left[ \begin{array}{cc} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{array}\right]
        \left[\begin{array}{rr} 1 & 0 & -1 \\ 0 & 1 & -2 \\ 0 & 0 & 1  \end{array}\right]
        = \left[ \begin{array}{cc} \cos\theta & -\sin\theta & -\cos\theta + 2\sin\theta + 1 \\ \sin\theta & \cos\theta & -2\cos\theta - \sin\theta + 2 \\ 0 & 0 & 1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$ Apply the transformation to the shape defined by $\texttt{coords}$ and plot the results.

        **Solution:**

        If we take $\theta = \frac{\pi}{4}$ then our rotation matrix becomes

        $$
        \begin{equation}
        RSL = \left[\begin{array}{rr} \frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2} & \frac{2 + \sqrt{2}}{2} \\ \frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2} & \frac{4 - 3\sqrt{2}}{2} \\ 0 & 0 & 1  \end{array}\right]
        \end{equation}
        $$

        To make it more obvious that this matrix transforms our vectors in the way that we want, we will write a function that applies the transformation and then plots the points several times.

        """
    )
    return


@app.cell
def _(np, plt, sqrt):
    coords_6 = np.array([[0, 0], [0.5, 0.5], [0.5, 1.5], [0, 1], [0, 0]])
    RSL = np.array([[sqrt(2) / 2, -sqrt(2) / 2, (2 + sqrt(2)) / 2], [sqrt(2) / 2, sqrt(2) / 2, (4 - 3 * sqrt(2)) / 2], [0, 0, 1]])

    def ApplyTransformation(coords, TransformationMatrix, NumOfApplications):
        coords = coords.transpose()
        x = coords[0, :]
        y = coords[1, :]
        Ones = np.ones((1, coords.shape[1]))
        coords = np.vstack((x, y, Ones))
        fig, ax = plt.subplots()
        for i in range(NumOfApplications + 1):
            coords_transformed = coords
            for j in range(i):
                coords_transformed = TransformationMatrix @ coords_transformed
            x_transformed = coords_transformed[0, :]
            y_transformed = coords_transformed[1, :]
            ax.plot(x_transformed, y_transformed, 'bo')
            ax.plot(x_transformed, y_transformed, 'b')
        ax.axvline(x=0, color='k', ls=':')
        ax.axhline(y=0, color='k', ls=':')
        ax.grid(True)
        ax.axis([-2, 4, -1, 5])
        ax.set_aspect('equal')
    ApplyTransformation(coords_6, RSL, 7)
    return ApplyTransformation, RSL, coords_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** 

        $(a)$ Find a matrix that represents clockwise rotation of $180^\circ$ about the point $(1,1)$ followed by a shift $3$ units to the right.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        In order to carry out the clockwise rotation about the point $(1,1)$, we first need to bring the center of rotation to $(0,0)$, for which we need to do *translation* involving the shift of $1$ unit left and $1$ unit down. Then, we need to do clockwise rotation with respect to $(0,0)$. After performing the rotation transformation, we need to redo the *translation* involving the shift of $1$ unit right and $1$ unit up so that the rotation is actually carried about the point $(1,1)$. After this whole rotation transformation is performed about the point (1,1), we need to carry out another *translation* transformation that involves a shift of 3 units to the right.


        Let us consider that the matrix $A$ represents the *translation* transformation to bring the center of rotation to $(0,0)$, matrix $R$ represents the clockwise transformation about $(0,0)$ , the matrix $C$ represents the *translation* transformation to take the center of rotation to $(1,1)$. Lastly, the matrix $D$ represents the *translation* transformation to shift $3$ units right.

        The matrices $A$, $C$ and $D$ are as follows:

        $$
        \begin{equation}
        A = \left[ \begin{array}{cc} 1 & 0 & -1\\ 0 & 1 & -1 \\ 0 & 0 & 1 \end{array}\right] \hspace{1cm}
        C = \left[ \begin{array}{cc} 1 & 0 & 1\\ 0 & 1 & 1 \\ 0 & 0 & 1 \end{array}\right] \hspace{1cm}
        D = \left[ \begin{array}{cc} 1 & 0 & 3\\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        The matrix $R$ representing clockwise rotation is as follows:

        $$
        \begin{equation}
        R = \left[ \begin{array}{cc} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{array}\right]
        \end{equation}
        $$


        A single matrix that represents clockwise rotation of $180^\circ$ about the point $(1,1)$ followed by a shift $3$ units to the right is given by the product $DCRA$.


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$  Apply the transformation to the shape defined by $\texttt{coords}$ and plot the results.
        """
    )
    return


@app.cell
def _(cos, np, pi, plt, sin):
    coords_7 = np.array([[0, 0], [0, 3], [1, 3], [1, 1], [2, 1], [2, 0], [0, 0]])
    coords_7 = coords_7.transpose()
    x_7 = coords_7[0, :]
    y_7 = coords_7[1, :]
    Ones_1 = np.ones((1, 7))
    coords_7 = np.vstack((x_7, y_7, Ones_1))
    theta_1 = pi
    A_2 = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]])
    R_2 = np.array([[cos(theta_1), sin(theta_1), 0], [-sin(theta_1), cos(theta_1), 0], [0, 0, 1]])
    C_2 = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
    D_1 = np.array([[1, 0, 3], [0, 1, 0], [0, 0, 1]])
    DCRA = D_1 @ C_2 @ R_2 @ A_2
    coords_translated = DCRA @ coords_7
    x_translated = coords_translated[0, :]
    y_translated = coords_translated[1, :]
    fig_7, ax_7 = plt.subplots()
    ax_7.plot(x_7, y_7, 'ro')
    ax_7.plot(x_translated, y_translated, 'bo')
    ax_7.plot(x_7, y_7, 'r', ls='--')
    ax_7.plot(x_translated, y_translated, 'b')
    ax_7.axvline(x=0, color='k', ls=':')
    ax_7.axhline(y=0, color='k', ls=':')
    ax_7.grid(True)
    ax_7.axis([-1, 6, -2, 6])
    ax_7.set_aspect('equal')
    ax_7.set_title('Translation')
    return (
        A_2,
        C_2,
        DCRA,
        D_1,
        Ones_1,
        R_2,
        ax_7,
        coords_7,
        coords_translated,
        fig_7,
        theta_1,
        x_7,
        x_translated,
        y_7,
        y_translated,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** 

        $(a)$ Find a single matrix that represents a transformation that has the effect of reflection about the $x_2$-axis followed by a shift $2$ units to the right and $2$ units up.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        In accordance with our discussion in this section, we need to view $\mathbb{R}^2$ as a plane within $\mathbb{R}^3$. Then, we can consider two linear transformations, $T:\mathbb{R}^3\to\mathbb{R}^3$ and $S:\mathbb{R}^3\to\mathbb{R}^3$. The linear transformation $T$ has the effect of reflection about the $x_2$-axis. $S$ has the *effect of translation* involving the shift of $2$ units to the right and $2$ units up.

        Let the matrix $A$ represents the reflection transformation and the matrix $B$ represents the *translation* transformation. Both the matrices $A$ and $B$ are $3 \times 3$ since the input and output spaces for both the transformations are $\mathbb{R}^3$.

        The matrices $A$ and $B$ are as follows:

        $$
        \begin{equation}
        A = \left[ \begin{array}{cc} -1 & 0 & 0\\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{array}\right] \hspace{1cm}
        B = \left[ \begin{array}{cc} 1 & 0 & 2\\ 0 & 1 & 2 \\ 0 & 0 & 1 \end{array}\right] 
        \end{equation}
        $$

        A single matrix that represents a transformation involving reflection followed by *translation* is given by the product $BA$.

        $$
        \begin{equation}
        BA = \left[ \begin{array}{cc} -1 & 0 & 2\\ 0 & 1 & 2 \\ 0 & 0 & 1 \end{array}\right] 
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$ Create a new matrix of coordinates, apply this transformation, and plot the results.
        """
    )
    return


@app.cell
def _(np, plt):
    coordinates = np.array([[0, 0], [0, 2], [3, 4], [2, 0], [0, 0]])
    coordinates = coordinates.transpose()
    x_8 = coordinates[0, :]
    y_8 = coordinates[1, :]
    Ones_2 = np.ones((1, 5))
    coordinates = np.vstack((x_8, y_8, Ones_2))
    BA = np.array([[-1, 0, 2], [0, 1, 2], [0, 0, 1]])
    coords_translated_1 = BA @ coordinates
    x_translated_1 = coords_translated_1[0, :]
    y_translated_1 = coords_translated_1[1, :]
    fig_8, ax_8 = plt.subplots()
    ax_8.plot(x_8, y_8, 'ro')
    ax_8.plot(x_translated_1, y_translated_1, 'bo')
    ax_8.plot(x_8, y_8, 'r', ls='--')
    ax_8.plot(x_translated_1, y_translated_1, 'b')
    ax_8.axvline(x=0, color='k', ls=':')
    ax_8.axhline(y=0, color='k', ls=':')
    ax_8.grid(True)
    ax_8.axis([-3, 6, -3, 10])
    ax_8.set_aspect('equal')
    ax_8.set_title('Reflection followed by translation')
    return (
        BA,
        Ones_2,
        ax_8,
        coordinates,
        coords_translated_1,
        fig_8,
        x_8,
        x_translated_1,
        y_8,
        y_translated_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Discrete Dynamical Systems
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** The webpages and their links are an example of a [directed graph](Applications.ipynb) as discussed in an earlier chapter.  An adjacency matrix therefore could be used to conveniently describe the link structure among the pages.  The matrix used in the web navigation model could then be constructed from the adjacency matrix.  Write a Python function that accepts an adjacency matrix for a direct graph, and returns the matrix required for the corresponding web navigation model.  Test your function on the following adjacency matrix defined in the cell below.
        """
    )
    return


@app.cell
def _(np):
    A_3 = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]])
    return (A_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The desired matrix in our web navigation model has probability $a_{ij}$ that the browser navigates from page $j$ to page $i$ but an adjacency matrix has $a_{ij} = 1$ if page $i$ has a link to page $j$, so for simplicity's sake we first take the transpose of $A$. We then loop over each column, counting how many pages site $j$ links to (call it $k$) then changing every 1 in the $j^{\text{th}}$ column to $\frac{0.8}{k} + \frac{0.2}{n-1}$ and every 0 in the $j^{\text{th}}$ column to $\frac{0.2}{n-1}$ (where $n \times n$ is the size of our adjacency matrix).
        """
    )
    return


@app.cell
def _(np):
    def ConvertAdjacencyMatrix(A):
        '''
        ConvertAdjacencyMatrix(A)
        
        ConvertAdjacencyMatrix converts an adjacency matrix into the matrix
        required for the corresponding web navigation model. There is no error
        checking to make sure that A is a proper adjacency matrix.

        Parameters
        ----------
        A : NumPy array object of dimension nxn

        Returns
        -------
        A_transpose : NumPy array object of dimension nxn
        '''
        
        A_transpose = np.transpose(A)
        n = A.shape[0]  # n is the number of rows and columns in A

        for j in range(0,n):                             
            k = 0                           # k counts the number of 1's in column j
            for i in range(0,n):
                if (A_transpose[i][j] == 1):
                    k = k + 1
            for i in range(0,n):
                if (A_transpose[i][j] == 1):
                    A_transpose[i][j] = 0.8/k + 0.2/(n-1)
                elif (i != j):
                    A_transpose[i][j] = 0.2/(n-1)
                
        return(A_transpose)
    return (ConvertAdjacencyMatrix,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We will now test our function on $A$, being careful not to round the changes to $\texttt{A_transpose}$.
        """
    )
    return


@app.cell
def _(ConvertAdjacencyMatrix, np):
    A_4 = np.array([[0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1, 1, 0, 0], [1, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 0, 0, 0, 0], [0, 1, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 1, 0, 1, 0, 1, 0, 0], [1, 0, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=float)
    print(np.round(ConvertAdjacencyMatrix(A_4), 3))
    return (A_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** There are a total of $4$ webpages in a web navigation model where page $1$ links to all other pages, page $2$ links to page $1$, page $3$ links to page $2$, page $4$ links to page $2$ and $3$. Create an adjacency matrix that describes the above link structure among the four pages. Use the Python function defined in the previous question to get the corresponding matrix used in the web navigation model. Given that  the browser starts the navigation at page $2$, predict the probability of browser vising each of the four pages after some large number of steps of navigation. Use your results to determine which page is most likely to be visited by the browser.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**


        The adjacency matrix $A$ describing the given link structure is as follows:

        $$
        \begin{equation}
        A = \left[ \begin{array}{ccccc} 
        0 & 1 & 0 & 0  \\ 
        1 & 0 & 1 & 1   \\
        1 & 0 & 0 & 1  \\
        1 & 0 & 0 & 0  \\
        \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let us see what the corresponding matrix used in the web navigation looks like:


        """
    )
    return


@app.cell
def _(ConvertAdjacencyMatrix, np):
    A_5 = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]], dtype=float)
    B_2 = ConvertAdjacencyMatrix(A_5)
    print('B: \n', B_2, '\n')
    X_3 = np.array([[0], [1], [0], [0]])
    n = 100
    for i in range(n):
        X_3 = B_2 @ X_3
    print('The vector representing probability of browser visiting each of the four pages is as follows: \n', X_3)
    return A_5, B_2, X_3, i, n


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        The computations show that page $1$ is most likely to be visited by the browser.  This makes sense since page $1$ has the most links to it.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
