import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Eigenvalues
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
    import math
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Determine the eigenvalues and corresponding eigenvectors of the following matrices by considering the transformations that they represent. Check your answers with a few computations.

        $(a)$

        $$
        \begin{equation}
        A = \left[ \begin{array}{cc} 1 & 0 \\ 0 & -3 \end{array}\right]
        \end{equation}
        $$

        **Solution:**

        Recall from [Chapter 4](Planar_Transformations.ipynb) that applying the transformation represented by $A$ is equivalent to stretching in the direction of the $y$-axis by a factor of 3 and then reflecting over the $y$-axis. If we imagine what this does to vectors in the plane then we might see that any vector that lies on the $x$-axis will be left unaffected. Therefore our first eigenvalue is $\lambda_1 = 1$ which corresponds to any scalar multiple of our first eigenvector $V_1 = \begin{equation} \left[ \begin{array}{cc} 1 \\ 0 \end{array}\right] \end{equation}$. Additionally, any vector that lies on the $y$-axis will simply be scaled by a factor of $-3$. Therefore our second eigenvalue is $\lambda_2 = -3$ which corresponds to any scalar multiple of our second eigenvector $V_2 = \begin{equation} \left[ \begin{array}{cc} 0 \\ 1 \end{array}\right] \end{equation}$.
        """
    )
    return


@app.cell
def _(np):
    A = np.array([[1,0],[0,-3]])
    V1 = np.array([[1],[0]])
    V2 = np.array([[0],[1]])
    R = np.array([[5],[0]])
    S = np.array([[1],[1]])
    T = np.array([[0],[0]])

    print(A@V1,'\n')
    print(A@V2,'\n')
    print(A@R,'\n')
    print(A@S,'\n')
    print(A@T)
    return A, R, S, T, V1, V2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$

        $$
        \begin{equation}
        B = \left[ \begin{array}{cc} 1 & 1 \\ 0 & 1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solutions:**

        Recall from [Chapter 4](Planar_Transformations.ipynb) that applying the transformation represented by $B$ is equivalent to shearing along the $x$-axis with a shearing factor of 1. Any vector that lies along the $x$-axis will be left unchanged because its $y$-coordinate is 0 and thus adds nothing to $x$-coordinate. Therefore our first eigenvalue is $\lambda_1 = 1$ which corresponds to any scalar multiple of our first eigenvector $V_1 = \begin{equation} \left[ \begin{array}{cc} 1 \\ 0 \end{array}\right] \end{equation}$. Any other vector with a nonzero $y$-coordinate will shear off of its original span, and thus cannot be a scalar multiple of itself. Therefore $V_1$ is the only eigenvalue of $B$.
        """
    )
    return


@app.cell
def _(np):
    B = np.array([[1, 1], [0, 1]])
    V1_1 = np.array([[1], [0]])
    V2_1 = np.array([[0], [1]])
    R_1 = np.array([[5], [0]])
    S_1 = np.array([[1], [1]])
    T_1 = np.array([[0], [0]])
    print(B @ V1_1, '\n')
    print(B @ V2_1, '\n')
    print(B @ R_1, '\n')
    print(B @ S_1, '\n')
    print(B @ T_1)
    return B, R_1, S_1, T_1, V1_1, V2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(c)$

        $$
        \begin{equation}
        C = \left[ \begin{array}{cc} cos(\frac{\pi}{2}) & -sin(\frac{\pi}{2}) \\ sin(\frac{\pi}{2}) & cos(\frac{\pi}{2}) \end{array}\right]
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

        Recall from [Chapter 4](Planar_Transformations.ipynb) that applying the transformation represented by $C$ is equivalent to a rotation about the origin by the angle $\frac{\pi}{2}$. Any nonzero vector that is transformed by this matrix will be pointing directly perpendicular to its original direction, so cannot lie on its original span. Therefore $C$ has no eigenvalues nor any eigenvectors.
        """
    )
    return


@app.cell
def _(math, np):
    C = np.array([[math.cos(math.pi / 2), -math.sin(math.pi / 2)], [math.sin(math.pi / 2), math.cos(math.pi / 2)]])
    V1_2 = np.array([[1], [0]])
    V2_2 = np.array([[0], [1]])
    R_2 = np.array([[5], [0]])
    S_2 = np.array([[1], [1]])
    T_2 = np.array([[0], [0]])
    print(np.round(C @ V1_2), '\n')
    print(np.round(C @ V2_2), '\n')
    print(np.round(C @ R_2), '\n')
    print(np.round(C @ S_2), '\n')
    print(np.round(C @ T_2))
    return C, R_2, S_2, T_2, V1_2, V2_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(d)$

        $$
        \begin{equation}
        D = \left[ \begin{array}{cc} -0.6 & -0.8 \\ -0.8 & 0.6 \end{array}\right]
        \end{equation}
        $$

        **Solution:** If we plot a shape before and after applying this transformation, it appears that this matrix reflects over the line $y=-2x$.
        """
    )
    return


@app.cell
def _(np, plt):
    coords = np.array([[0,0],[1,0],[1,1],[0,1],[0,0]])
    coords = coords.transpose()
    D = np.array([[-0.6,-0.8],[-0.8,0.6]])
    D_coords = D@coords

    x = coords[0,:]
    y = coords[1,:]
    x_LT = D_coords[0,:]
    y_LT = D_coords[1,:]

    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Plot the points and the line y = -2x. x and y are original vectors, x_LT and y_LT are images
    ax.plot(x,y,'ro')
    ax.plot(x_LT,y_LT,'bo')
    a=np.linspace(-2,2,100)
    ax.plot(a,-2*a,ls=':')

    # Connect the points by lines
    ax.plot(x,y,'r',ls="--")
    ax.plot(x_LT,y_LT,'b')

    # Edit some settings 
    ax.axvline(x=0,color="k",ls=":")
    ax.axhline(y=0,color="k",ls=":")
    ax.grid(True)
    ax.axis([-2,2,-1,2])
    ax.set_aspect('equal')
    return D, D_coords, a, ax, coords, fig, x, x_LT, y, y_LT


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Any vector that lies on the line $y = -2x$ will be left unchanged, so our first eigenvalue is $\lambda_1 = 1$ which corresponds to any scalar multiple of our first eigenvector $V_1 = \begin{equation} \left[ \begin{array}{cc} 1 \\ -2 \end{array}\right] \end{equation}$. Any vector that lies on the line orthogonal to the first, namely $y = \frac{1}{2}x$ will be simply flipped in direction, or in other words scaled by a factor of $-1$. Thus our second eigenvalue is $\lambda_2 = -1$ which corresponds to any scalar multiple of our second eigenvector $V_2 = \begin{equation} \left[ \begin{array}{cc} 2 \\ 1 \end{array}\right] \end{equation}$.
        """
    )
    return


@app.cell
def _(np):
    D_1 = np.array([[-0.6, -0.8], [-0.8, 0.6]])
    V1_3 = np.array([[1], [-2]])
    V2_3 = np.array([[2], [1]])
    R_3 = np.array([[5], [0]])
    S_3 = np.array([[1], [1]])
    T_3 = np.array([[0], [0]])
    print(D_1 @ V1_3, '\n')
    print(D_1 @ V2_3, '\n')
    print(D_1 @ R_3, '\n')
    print(D_1 @ S_3, '\n')
    print(D_1 @ T_3)
    return D_1, R_3, S_3, T_3, V1_3, V2_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Find the eigenvalues and the corresponding eigenvectors of the following matrix $R$ that represents the reflection transformation about the line $ x_1 = x_2 $.


        $$
        \begin{equation}
        R = \left[ \begin{array}{cc} 0 & 1 \\ 1 & 0 \end{array}\right]
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

        Let us consider the vector $ V_1$ as follows:


        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1  \\ 1 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        We can see that $ RV_1 = V_1$. So, $\lambda_1 = 1$ is one of the eigenvalues of $R$. 

        Similarly, let us consider the vector $V_2$ as follows:

        $$
        \begin{equation}
        V_2 = \left[ \begin{array}{r} -1 \\ 1 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        We observe that $RV_2 = -V_2$. 

        So, $\lambda_2 = -1$ is another eigenvalue of $R$.

        Therefore, we can say that there are two eigenvalues of the matrix $R$ i.e $\lambda_1 = 1$, $\lambda_2 = -1$. The vector $V_1$ and all its scalar multiples are eigenvectors associated with $\lambda_1$, $V_2$ and all its scalar multiples are eigenvectors associated with $\lambda_2$.
        """
    )
    return


@app.cell
def _(np):
    V_1 = np.array([[1], [1]])
    V_2 = np.array([[1], [-1]])
    R_4 = np.array([[0, 1], [1, 0]])
    print('RV_1: \n', R_4 @ V_1, '\n')
    print('RV_2: \n', R_4 @ V_2, '\n')
    return R_4, V_1, V_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Find a matrix that represents a vertical and horizontal stretch by a factor of $2$. Then, find the eigenvalues and the eigenvectors associated with those eigenvalues. (You may have to take a look at the Planar Transformations section  [Planar Transformations](Planar_Transformations.ipynb)).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Based on our discussion on Planar Transformations, the matrix that represents a vertical stretch and horizontal stretch is as follows:


        $$
        \begin{equation}
        A = \left[ \begin{array}{cc} 2 & 0 \\ 0 & 2 \end{array}\right]
        \end{equation}
        $$

        Since the effect of the transformation is to scale all vectors by a factor of $2$, *all vectors* are eigenvectors of the matrix $A$.  

        Let us consider the vector $V_1$ as an example.


        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1  \\ 1 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        We can see that $AV_1 = 2V_1$.  The only eigenvalue of $A$ is $2$.

        """
    )
    return


@app.cell
def _(np):
    A_1 = np.array([[2, 0], [0, 2]])
    V_1_1 = np.array([[1], [1]])
    V_2_1 = np.array([[1], [0]])
    print('AV_1: \n', A_1 @ V_1_1, '\n')
    print('AV_2: \n', A_1 @ V_2_1, '\n')
    return A_1, V_1_1, V_2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Find a matrix that represents reflection about the $x_1-$axis and find its eigenvalues and eigenvectors.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Based on our discussion on Planar Transformations, the matrix that represents reflection about the $x_1$ -axis is as follows:


        $$
        \begin{equation}
        B = \left[ \begin{array}{cc} 1 & 0 \\ 0 & -1 \end{array}\right]
        \end{equation}
        $$


        Let us consider the vector $V_1$ as follows:


        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1  \\ 0 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$


        Since $BV_1 = V_1$, one of the eigenvalues of the matrix $B$ is $\lambda_1  = 1$ and $V_1$ is the eigenvector associated with $\lambda_1$. Infact, all the scalar multiples of $V_1$ are also eigenvectors associated with $\lambda_1$.


        Now, let us consider another vector $V_2$ as follows:


        $$
        \begin{equation}
        V_2 = \left[ \begin{array}{r} 0  \\ 1 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        Since $BV_2 = -V_2$, other eigenvalue for the matrix $B$ will be $\lambda_2 =-1$ and $V_2$ is the eigenvector associated with $\lambda_2$. All the scalar multiples of $V_2$ are also eigenvectors associated with $\lambda_2$.                                      
        """
    )
    return


@app.cell
def _(np):
    B_1 = np.array([[1, 0], [0, -1]])
    V_1_2 = np.array([[1], [0]])
    V_2_2 = np.array([[0], [1]])
    print('BV_1: \n', B_1 @ V_1_2, '\n')
    print('BV_2: \n', B_1 @ V_2_2, '\n')
    return B_1, V_1_2, V_2_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Diagonalization

        **Exercise 1:** Find the diagonalization of the matrix from **Example 1**.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rr} 2 & -1  \\ -1 & 2 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:** Recall that $A$ has eigenvalues $\lambda_1 = 3$ and $\lambda_2 = 1$, and corresponding eigenvectors $V_1$ and $V_2$ defined below.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ -1 \end{array}\right] \hspace{2cm} V_2 = \left[ \begin{array}{r} 1 \\ 1 \end{array}\right]
        \end{equation}
        $$

        To find the diagonalization of $A$, we need to find the diagonal matrix $D$ with the eigenvalues of $A$ along its diagonal, and the matrix $S$ which has columns equal to the eigenvectors of $A$. We write them out below, and then verify that $A = SDS^{-1}$.

        $$
        \begin{equation}
        D = \left[ \begin{array}{r} 3 & 0 \\ 0 & 1 \end{array}\right] \hspace{2cm} S = \left[ \begin{array}{r} 1 & 1 \\ -1 & 1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    A_2 = np.array([[2, -1], [-1, 2]])
    D_2 = np.array([[3, 0], [0, 1]])
    S_4 = np.array([[1, 1], [-1, 1]])
    print(A_2 - S_4 @ D_2 @ lag.Inverse(S_4))
    return A_2, D_2, S_4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Find the diagonalization of the following matrix.

        $$
        \begin{equation}
        B = \left[ \begin{array}{rrr} 2 & 0 & 0  \\ 3 & -2 & 1 \\  1 & 0 & 1\end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:** First we must find the eigenvectors of $B$ and their corresponding eigenvalues. We make use of the $\texttt{eig}$ function to do this.
        """
    )
    return


@app.cell
def _(np, sla):
    B_2 = np.array([[2, 0, 0], [3, -2, 1], [1, 0, 1]])
    evalues, evectors = sla.eig(B_2)
    print(evalues, '\n')
    print(evectors)
    return B_2, evalues, evectors


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $S$ is equivalent to the matrix of eigenvectors above, so all we have to do is write out $D$ and then verify that $B = SDS^{-1}$.
        """
    )
    return


@app.cell
def _(B_2, evectors, lag, np):
    D_3 = np.array([[-2, 0, 0], [0, 1, 0], [0, 0, 2]])
    S_5 = evectors
    print(B_2 - S_5 @ D_3 @ lag.Inverse(S_5))
    return D_3, S_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Write a function that accepts an $n\times n$ matrix $A$ as an argument, and returns the three matrices $S$, $D$, and $S^{-1}$ such that $A=SDS^{-1}$.  Make use of the $\texttt{eig}$ function in SciPy.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:** We write a function that finds the diagonalization of an $n \times n$ matrix $A$ and then test it on the matrix $B$ defined in exercise 2 to check that it works.
        """
    )
    return


@app.cell
def _(lag, np, sla):
    def Diagonalization(A):
        """
        Diagonalization(A)
        
        Diagonalization takes an nxn matrix A and computes the diagonalization
        of A. There is no error checking to ensure that the eigenvectors of A 
        form a linearly independent set.

        Parameters
        ----------
        A : NumPy array object of dimension nxn

        Returns
        -------
        S :        NumPy array object of dimension nxn
        D :        NumPy array object of dimension nxn
        S_inverse: Numpy array object of dimension nxn
        """
        n = A.shape[0]
        D = np.zeros((3, 3), dtype='complex128')
        evalues, evectors = sla.eig(A)
        S = evectors
        S_inverse = lag.Inverse(S)
        for i in range(0, n, 1):
            D[i][i] = evalues[i]
        return (S, D, S_inverse)
    B_3 = np.array([[2, 0, 0], [3, -2, 1], [1, 0, 1]])
    S_6, D_4, S_inverse = Diagonalization(B_3)
    print(S_6, '\n')
    print(D_4, '\n')
    print(S_inverse)
    return B_3, D_4, Diagonalization, S_6, S_inverse


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Construct a $3\times 3$ matrix that is not diagonal and has eigenvalues 2, 4, and 10. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:** If we want to construct a matrix $C$ with eigenvalues 2, 4, and 10, then $D$ must be a diagonal matrix with those eigenvalues as its diagonal entries. If we let $S$ be any $3 \times 3$ non diagonal matrix with linearly independent columns, then $C = SDS^{-1}$ will give us our desired matrix. We define $D$ and $S$ below, calculate $C$, and then check our answer using the $\texttt{eig}$ function.

        $$
        \begin{equation}
        D = \left[ \begin{array}{rrr} 2 & 0 & 0  \\ 0 & 4 & 0 \\  0 & 0 & 10\end{array}\right] \hspace{2cm} S = \left[ \begin{array}{rrr} 1 & 1 & 0  \\ 0 & 1 & 1 \\  1 & 0 & 1\end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np, sla):
    D_5 = np.array([[2, 0, 0], [0, 4, 0], [0, 0, 10]])
    S_7 = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    C_1 = S_7 @ D_5 @ lag.Inverse(S_7)
    print(C_1, '\n')
    evalues_1, evectors_1 = sla.eig(C_1)
    print(evalues_1)
    return C_1, D_5, S_7, evalues_1, evectors_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The eigenvalues determined by $\texttt{eig}$ appear in a different order than we had originally, but are correct.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Suppose that $C = QDQ^{-1}$ where

        $$
        \begin{equation}
        Q = \left[ \begin{array}{c|c|c} & & \\ U & V & W \\ & & \end{array} \right] \hspace{2cm} D = \left[ \begin{array}{rrr} 0 & 0 & 0  \\ 0 & 3 & 0 \\  0 & 0 & 5 \end{array}\right]
        \end{equation}
        $$

        $(a)$ Give a basis for $\mathcal{N}(C)$

        **Solution:** Consider the set $\{U,V,W\}$. This set of three vectors is linearly independent, so it forms a basis for $\mathbb{R}^3$ and thus any vector $X \in \mathbb{R}^3$ can be expressed as a linear combination $X = aU + bV +cW$ for some scalars $a,b,$ and $c$. Then we have

        $$
        \begin{equation}
        CX \hspace{0.2cm} = \hspace{0.2cm} C(aU + bV + cW) \hspace{0.2cm} = \hspace{0.2cm} aCU + bCV + cCW \hspace{0.2cm} = \hspace{0.2cm} b3V + c5W
        \end{equation}
        $$

        Since $V$ and $W$ are linearly independent, so are $3V$ and $5W$, and therefore the only way that $b3V + c5W = 0$ is if $b = c = 0$. This means that $CX = 0$ when $b = c = 0$ and there is no restriction on $a$. But then any element in the null space of $C$ can be expressed as a scalar multiple of $U$ and thus $\{U\}$ is a basis for $\mathcal{N}(C)$.

        $(b)$ Find all the solutions to $CX = V + W$

        **Solution:** Let $X$ be an arbitrary vector in $\mathbb{R}^3$. Since the set of three vectors $\{U,V,W\}$ are linearly independent, they form a basis for $\mathbb{R}^3$, and thus we can write $X = aU + bV + cW$ for some scalars $a,b,$ and $c$. Then we have

        $$
        \begin{equation}
        CX \hspace{0.2cm} = \hspace{0.2cm} C(aU + bV + cW) \hspace{0.2cm} = \hspace{0.2cm} aCU + bCV + cCW \hspace{0.2cm} = \hspace{0.2cm} b3V + c5W
        \end{equation}
        $$

        This whole thing is equal to $V+W$ when $b = \frac{1}{3}$, $c = \frac{1}{5}$, and is independent of $a$. Therefore the solutions to $CX = V + W$ are the vectors of the form $X = aU + \frac{1}{3}V + \frac{1}{5}W$ where $a$ is any real number.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Approximating Eigenvalues

        **Exercise 1:** Let $A$ be the matrix from the Inverse Power Method example.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr} 9 & -1 & -3 \\ 0 & 6 & 0 \\ -6 & 3 & 6 \end{array}\right]
        \end{equation}
        $$

        $(a)$ Use the Power Method to approximate the largest eigenvalue $\lambda_1$.  Verify that the exact value of $\lambda_1$ is 12.

        **Solution:**
        """
    )
    return


@app.cell
def _(lag, np):
    A_3 = np.array([[9, -1, -3], [0, 6, 0], [-6, 3, 6]])
    X = np.array([[0], [1], [0]])
    m = 0
    tolerance = 0.0001
    MAX_ITERATIONS = 100
    Y = A_3 @ X
    difference = Y - lag.Magnitude(Y) * X
    while m < MAX_ITERATIONS and lag.Magnitude(difference) > tolerance:
        X = Y
        X = X / lag.Magnitude(X)
        Y = A_3 @ X
        difference = Y - lag.Magnitude(Y) * X
        m = m + 1
    print('Eigenvector is approximately:')
    print(X, '\n')
    print('Magnitude of the eigenvalue is approximately:')
    print(lag.Magnitude(Y), '\n')
    return A_3, MAX_ITERATIONS, X, Y, difference, m, tolerance


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It appears that the first entry of $X$ is equal to the third multiplied by $-1$. If we set the first entry to be 1 and the third entry to be -1, we can calculate the following product and see that $\lambda_1$ is indeed 12.

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 9 & -1 & -3 \\ 0 & 6 & 0 \\ -6 & 3 & 6 \end{array}\right]
        \left[ \begin{array}{r} 1\\ 0 \\ -1 \end{array}\right] =
        \left[ \begin{array}{r} 12\\ 0 \\ -12 \end{array}\right] = 12X
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$ Apply the Inverse Power Method with a shift of $\mu = 10$.  Explain why the results differ from those in the example.

        **Solution:**
        """
    )
    return


@app.cell
def _(lag, np, sla):
    A_4 = np.array([[9, -1, -3], [0, 6, 0], [-6, 3, 6]])
    X_1 = np.array([[0], [1], [0]])
    m_1 = 0
    tolerance_1 = 0.0001
    MAX_ITERATIONS_1 = 100
    difference_1 = X_1
    A_4 = np.array([[9, -1, -3], [0, 6, 0], [-6, 3, 6]])
    I = np.eye(3)
    mu = 10
    Shifted_A = A_4 - mu * I
    LU_factorization = sla.lu_factor(Shifted_A)
    while m_1 < MAX_ITERATIONS_1 and lag.Magnitude(difference_1) > tolerance_1:
        X_previous = X_1
        X_1 = sla.lu_solve(LU_factorization, X_1)
        X_1 = X_1 / lag.Magnitude(X_1)
        difference_1 = X_1 - X_previous
        m_1 = m_1 + 1
    print('Eigenvector is approximately:')
    print(X_1, '\n')
    print('Eigenvalue of A is approximately:')
    print(lag.Magnitude(A_4 @ X_1))
    return (
        A_4,
        I,
        LU_factorization,
        MAX_ITERATIONS_1,
        Shifted_A,
        X_1,
        X_previous,
        difference_1,
        m_1,
        mu,
        tolerance_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Inverse Power Method with a shift of $\mu$ finds the eigenvalue closest to $\mu$. $A$ has three eigenvalues, namely 3, 6, and 12, of which 10 is closest to 12 and 7.5 is closest to 6. This is why when we used $\lambda = 7.5$ in the example we got the eigenvalue 6, but when we used $\lambda = 10$ we got the eigenvalue 12. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(c)$ Apply the Inverse Power Method with a shift of $\mu = 7.5$ and the initial vector given below.  Explain why the sequence of vectors approach the eigenvector corresponding to $\lambda_1$

        $$
        \begin{equation}
        X^{(0)} = \left[ \begin{array}{r} 1 \\ 0  \\ 0 \end{array}\right]
        \end{equation}
        $$

        **Solution:**
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Applications

        **Exercise 1:** Experiment with a range of initial conditions in the infectious disease model to provide evidence that an equilibrium state is reached for all meaningful initial states. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We will create a method that uses the model given in the example but takes in a vector representing the initial conditions, then outputs the graphs of the values over time. We will input several different sets of intial conditions and graph them, verifying that they all seem to converge upon the same equilibrium state.
        """
    )
    return


@app.cell
def _(np, plt):
    def Iterate_SIRS(X):

        A = np.array([[0.95, 0, 0.15],[0.05,0.8,0],[0,0.2,0.85]])

        ## T is final time
        T = 20

        ## The first column of results contains the initial values 
        results = np.copy(X)

        for i in range(T):
            X = A@X
            results = np.hstack((results,X))

        ## t contains the time indices 0, 1, 2, ..., T
        t = np.linspace(0,T,T+1)
        ## s, i, r values are the rows of the results array
        s = results[0,:]
        i = results[1,:]
        r = results[2,:]

        fig,ax = plt.subplots()

        ## The optional label keyword argument provides text that is used to create a legend
        ax.plot(t,s,'b+',label="Susceptible");
        ax.plot(t,i,'rx',label="Infectious");
        ax.plot(t,r,'g+',label="Recovered");

        ax.set_ylim(0,1.1)
        ax.grid(True)
        ax.legend();
        
    Iterate_SIRS(np.array([[0.95],[0.05],[0]]))
    Iterate_SIRS(np.array([[0.7],[0.05],[0.25]]))
    Iterate_SIRS(np.array([[0.3],[0.4],[0.3]]))
    Iterate_SIRS(np.array([[0.1],[0.8],[0.1]]))
    return (Iterate_SIRS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The values of $s_{20}, i_{20},$ and $r_{20}$ see to be nearly equal, so it appears that regardless of the initial conditions, the SIRS model always converges to the same equilibrium state.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Perform an analysis similar to that in the example for the following infectious disease model.  In this model the rate at which individuals move from the Recovered category to the Susceptible category is less than that in the example.  Make a plot similar to that in the example and also calculate the theoretical equilibrium values for $s$, $i$, and $r$.

        $$
        \begin{equation}
        X_t = \left[ \begin{array}{r} s_t \\ i_t \\ r_t  \end{array}\right] =
        \left[ \begin{array}{rrr} 0.95 & 0 & 0.05 \\ 0.05 & 0.80 & 0 \\ 0 & 0.20 & 0.95 \end{array}\right]
        \left[ \begin{array}{r} s_{t-1} \\ i_{t-1} \\ r_{t-1}  \end{array}\right]=
        AX_{t-1}
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We will plot the first 50 iterations of $s_t,i_t,$ and $r_t$ using the same initial conditions as the example.
        """
    )
    return


@app.cell
def _(np, plt):
    A_5 = np.array([[0.95, 0, 0.05], [0.05, 0.8, 0], [0, 0.2, 0.95]])
    T_4 = 50
    X_2 = np.array([[0.95], [0.05], [0]])
    results = np.copy(X_2)
    for i in range(T_4):
        X_2 = A_5 @ X_2
        results = np.hstack((results, X_2))
    t = np.linspace(0, T_4, T_4 + 1)
    s = results[0, :]
    i = results[1, :]
    r = results[2, :]
    fig_1, ax_1 = plt.subplots()
    ax_1.plot(t, s, 'b+', label='Susceptible')
    ax_1.plot(t, i, 'rx', label='Infectious')
    ax_1.plot(t, r, 'g+', label='Recovered')
    ax_1.set_ylim(0, 1.1)
    ax_1.grid(True)
    ax_1.legend()
    return A_5, T_4, X_2, ax_1, fig_1, i, r, results, s, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It  appears that the population has reached an equilibrium after about 45 weeks, but one that was different than in the example. The proportion of the population that is susceptible seems to be equal to the porportion of the population that is recovered at roughly 45%, and the proportion that is infectious seems to be about 10%. To find the true equilibrium values, we calculate the $\texttt{RREF}$ of the augmented matrix $[(A-I)|0]$.
        """
    )
    return


@app.cell
def _(A_5, lag, np):
    I_1 = np.eye(3)
    ZERO = np.zeros((3, 1))
    augmented_matrix = np.hstack((A_5 - I_1, ZERO))
    reduced_matrix = lag.FullRowReduction(augmented_matrix)
    print(reduced_matrix)
    return I_1, ZERO, augmented_matrix, reduced_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the reduced system for the equilibrium values $s$, $i$, and $r$, we can take $r$ as the free variable and write $s=r$ and $i=0.25r$.  For any value of $r$, a vector of the following form is an eigenvector for $A-I$, corresponding to the eigenvalue one.

        $$
        \begin{equation}
        r\left[ \begin{array}{r} 1 \\ 0.25 \\ 1  \end{array}\right]
        \end{equation}
        $$

        If we add the constraint that $s + i + r = 1$, then we get the equation $r + 0.25r + r = 1$ which gives the unique equilibrium values of $r = 4/9$, $s=4/9$, and $i=1/9$.  If we carry out a large number of iterations, we see that the computed values are very close to the theoretical equilibrium values. 
        """
    )
    return


@app.cell
def _(A_5, np):
    T_5 = 100
    X_3 = np.array([[0.95], [0.05], [0]])
    for i_1 in range(T_5):
        X_3 = A_5 @ X_3
    print('Computed values of s, i, r at time ', T_5, ':')
    print(X_3)
    return T_5, X_3, i_1


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
