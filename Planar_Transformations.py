import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Transformations in a Plane
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As discussed in the previous section, any linear transformation $T:\mathbb{R}^2\to\mathbb{R}^2$ can be represented as the multiplication of a $2\times 2$ matrix and a coordinate vector.  For now, the coordinates are with respect to the standard basis $\{E_1,E_2\}$ as they were before.  The columns of the matrix are then the images of the the basis vectors.

        $$
        \begin{equation}
        A = \left[ \begin{array}{c|c} T(E_1) & T(E_2) \end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1:  Horizontal Stretch

        As a first example let us consider the transformation defined by the following images.

        $$
        \begin{equation}
        T\left(\left[ \begin{array}{r} 1 \\ 0  \end{array}\right]\right)= \left[ \begin{array}{r} 2 \\ 0  \end{array}\right] \hspace{1in}  T\left(\left[ \begin{array}{r} 0 \\ 1  \end{array}\right]\right)= \left[ \begin{array}{r} 0 \\ 1  \end{array}\right]  
        \end{equation}
        $$

        The matrix corresponding to the transformation is then

        $$
        \begin{equation}
        A = \left[ \begin{array}{cc} 2 & 0 \\ 0 & 1 \end{array}\right]
        \end{equation}
        $$

        The image of a general vector with components $c_1$ and $c_2$ is given by a linear combination 
        of the images of the basis vectors.

        $$
        \begin{equation}
        T\left(\left[ \begin{array}{r} c_1 \\ c_2  \end{array}\right]\right) = T\left(c_1\left[ \begin{array}{r} 1 \\ 0  \end{array}\right] + c_2 \left[ \begin{array}{r} 0 \\ 1  \end{array}\right]\right) = c_1T\left(\left[ \begin{array}{r} 1 \\ 0  \end{array}\right]\right) + c_2 T\left(\left[ \begin{array}{r} 0 \\ 1  \end{array}\right]\right)  = c_1\left[ \begin{array}{r} 2 \\ 0  \end{array}\right] + c_2\left[ \begin{array}{r} 0 \\ 1  \end{array}\right]  = \left[ \begin{array}{cc} 2 & 0 \\ 0 & 1 \end{array}\right] \left[\begin{array}{c} c_1 \\ c_2 \end{array}\right]
        \end{equation}
        $$

        To get a visual understanding of the transformation, we produce a plot that displays multiple input vectors, and their corresponding images.  Since we will multiply each vector by the same matrix, we can organize our calculations by constructing a matrix with columns that match the input vectors.  This matrix is the array labeled $\texttt{coords}$ in the code below.  We use array slicing to label the first row $\texttt{x}$, and the second row $\texttt{y}$ for the purposes of plotting. 
        """
    )
    return


@app.cell
def _():
    # '%matplotlib inline  \nfrom math import pi, sin, cos\nimport matplotlib.pyplot as plt\nimport numpy as np\n\ncoords = np.array([[0,0],[0.5,0.5],[0.5,1.5],[0,1],[0,0]])\ncoords = coords.transpose()\ncoords\nx = coords[0,:]\ny = coords[1,:]\n\nA = np.array([[2,0],[0,1]])\nA_coords = A@coords' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The columns of the array $\texttt{A_coords}$ are the images of the vectors that make up the columns of $\texttt{coords}$.  Let's plot the coordinates and see the effect of the transformation.  Again, we need to access the first and second rows by slicing.  The red points show the original coordinates, and the new coordinates are shown in blue.
        """
    )
    return


@app.cell
def _(A_coords, plt, x, y):
    x_LT1 = A_coords[0, :]
    y_LT1 = A_coords[1, :]
    _fig, _ax = plt.subplots()
    _ax.plot(x, y, 'ro')
    _ax.plot(x_LT1, y_LT1, 'bo')
    _ax.plot(x, y, 'r', ls='--')
    _ax.plot(x_LT1, y_LT1, 'b')
    _ax.axvline(x=0, color='k', ls=':')
    _ax.axhline(y=0, color='k', ls=':')
    _ax.grid(True)
    _ax.axis([-2, 2, -1, 2])
    _ax.set_aspect('equal')
    _ax.set_title('Horizontal Stretch')
    return x_LT1, y_LT1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We observe that the result of the transformation is that the polygon has been stretched in the horizontal direction.  

        ### Example 2:  Reflection

        Now that we have a set of vectors stored in the $\texttt{coords}$ matrix, let's look at another transformation that is defined by the matrix $B$.

        $$
        \begin{equation}
        B = \left[ \begin{array}{cc} -1 & 0 \\ 0 & 1 \end{array}\right]
        \end{equation}
        $$

        We make a new array $B$, but reuse the plotting code from the previous example.
        """
    )
    return


@app.cell
def _(coords, np, plt, x, y):
    B = np.array([[-1, 0], [0, 1]])
    B_coords = B @ coords
    x_LT2 = B_coords[0, :]
    y_LT2 = B_coords[1, :]
    _fig, _ax = plt.subplots()
    _ax.plot(x, y, 'ro')
    _ax.plot(x_LT2, y_LT2, 'bo')
    _ax.plot(x, y, 'r', ls='--')
    _ax.plot(x_LT2, y_LT2, 'b')
    _ax.axvline(x=0, color='k', ls=':')
    _ax.axhline(y=0, color='k', ls=':')
    _ax.grid(True)
    _ax.axis([-2, 2, -1, 2])
    _ax.set_aspect('equal')
    _ax.set_title('Reflection')
    return B, B_coords, x_LT2, y_LT2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that this transformation reflects points around the vertical axis.

        ### Example 3:  Rotation

        To rotate vectors in the plane, we choose an angle $\theta$ and write down the matrix that represents the rotation counterclockwise by an angle $\theta$.  Basic trigonometry can be used to calculate the columns in this case.   

        $$
        \begin{equation}
        R = \left[ \begin{array}{cc} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(coords, cos, np, pi, plt, sin, x, y):
    theta = pi / 6
    R = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    R_coords = R @ coords
    x_LT3 = R_coords[0, :]
    y_LT3 = R_coords[1, :]
    _fig, _ax = plt.subplots()
    _ax.plot(x, y, 'ro')
    _ax.plot(x_LT3, y_LT3, 'bo')
    _ax.plot(x, y, 'r', ls='--')
    _ax.plot(x_LT3, y_LT3, 'b')
    _ax.axvline(x=0, color='k', ls=':')
    _ax.axhline(y=0, color='k', ls=':')
    _ax.grid(True)
    _ax.axis([-2, 2, -1, 2])
    _ax.set_aspect('equal')
    _ax.set_title('Rotation')
    return R, R_coords, theta, x_LT3, y_LT3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 4: Shear

        In the study of mechanics, shear forces occur when one force acts on some part of a body while a second force acts on another part of the body but in the opposite direction. To visualize this, imagine a deck of playing cards resting on the table, and then while resting your hand on top of the deck, sliding your hand parallel to the table. 

        A **shear** transformation is any transformation that shifts points in a given direction by an amount proportional to their original (signed) distance from the line that is parallel to the movement of direction and that passes through the origin. For example, a horizontal shear applied to a vector would add to the vector's $x$-coordinate a value scaled by the value of its $y$-coordinate (or subtract, if the $y$-coordinate was negative). A vertical shear is expressed as a matrix in the form of the first matrix below, and a horizontal shear with the second where $k \in \mathbb{R}$ is called the **shearing factor**.

        $$
        \begin{equation}
        \left[ \begin{array}{r} 1 & 0 \\ k & 1  \end{array}\right] \hspace{1in}  
        \left[ \begin{array}{r} 1 & k \\ 0 & 1  \end{array}\right]  
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(coords, np, plt, x, y):
    S = np.array([[1, 2], [0, 1]])
    S_coords = S @ coords
    x_LT4 = S_coords[0, :]
    y_LT4 = S_coords[1, :]
    _fig, _ax = plt.subplots()
    _ax.plot(x, y, 'ro')
    _ax.plot(x_LT4, y_LT4, 'bo')
    _ax.plot(x, y, 'r', ls='--')
    _ax.plot(x_LT4, y_LT4, 'b')
    _ax.axvline(x=0, color='k', ls=':')
    _ax.axhline(y=0, color='k', ls=':')
    _ax.grid(True)
    _ax.axis([-2, 4, -1, 2])
    _ax.set_aspect('equal')
    _ax.set_title('Shear')
    return S, S_coords, x_LT4, y_LT4


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 5:  Composition of transformations

        One of the powerful aspects of representing linear transformations as matrices is that we can form compositions by matrix multiplication.  For example,  in order to find the matrix that represents the composition $B\circ R$, the rotation from Example 3, followed by the reflection of Example 2,
        we can simply multiply the matrices that represent the individual transformations.  

        $$
        \begin{equation}
        [B][R] = \left[ \begin{array}{cc} -1 & 0 \\ 0 & 1 \end{array}\right]
        \left[ \begin{array}{cc} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{array}\right]
        =\left[ \begin{array}{cc} -\cos\theta & \sin\theta \\ \sin\theta & \cos\theta \end{array}\right]
        \end{equation}
        $$

        """
    )
    return


@app.cell
def _(coords, cos, np, plt, sin, theta, x, y):
    C = np.array([[-cos(theta), sin(theta)], [sin(theta), cos(theta)]])
    C_coords = C @ coords
    x_LT5 = C_coords[0, :]
    y_LT5 = C_coords[1, :]
    _fig, _ax = plt.subplots()
    _ax.plot(x, y, 'ro')
    _ax.plot(x_LT5, y_LT5, 'bo')
    _ax.plot(x, y, 'r', ls='--')
    _ax.plot(x_LT5, y_LT5, 'b')
    _ax.axvline(x=0, color='k', ls=':')
    _ax.axhline(y=0, color='k', ls=':')
    _ax.grid(True)
    _ax.axis([-2, 2, -1, 2])
    _ax.set_aspect('equal')
    return C, C_coords, x_LT5, y_LT5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises
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


@app.cell
def _():
    ## Code solution here.
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
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** 

        ($a$) Find a matrix that represents the reflection about the line $x_1=x_2$. 
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
        ($b$) Multiply the matrix by $\texttt{coords}$ and plot the results.
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
        **Exercise 3:** 

        ($a$) Find a matrix that represents the rotation clockwise by an angle $\theta$. 
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
        ($b$) Let $\theta = 90^{\circ}$.  Multiply the matrix by $\texttt{coords}$ and plot the results.
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
        **Exercise 4:** 

        ($a$) Find a matrix that represents a vertical shear followed by the rotation in Example 3. 
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
        ($b$) Multiply the matrix by $\texttt{coords}$ and plot the results.
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
        **Exercise 5:** Create a new matrix of coordinates and apply one of the transformations in the Examples.  Plot the results.
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
        **Exercise 6:** 

        ($a$) Construct a matrix that represents a horizontal and a vertical stretch by a factor of 2 . 
        """
    )
    return


@app.cell
def _():
    ## Code solution here
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
def _():
    ## Code solution here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
