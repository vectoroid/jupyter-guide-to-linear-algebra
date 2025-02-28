import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Vector Spaces
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This chapter contain several concepts related to the the idea of a vector space.  The main objective of the chapter is to tie these concepts to linear systems.

        A **vector space** is a collection of objects, called vectors, together with definitions that allow for the addition of two vectors and the multiplication of a vector by a scalar.  These operations produce other vectors in the collection and they satisfy a list of algebraic requirements such as associativity and commutativity.  Although we will not consider the implications of each requirement here, we provide the list for reference.
        """
    )
    return


@app.cell
def _(mo):
    # Cell tags: hide-output
    mo.md(
        r"""
        For any vectors $U$, $V$, and $W$, and scalars $p$ and $q$, the definitions of vector addition and scalar multiplication for a vector space have the following properties:

        1. $U+V = V+U$ 
        2. $(U+V)+W = U+(V+W)$
        3. $U + 0 = U$ 
        4. $U + (-U) = 0$
        5. $p(qU) = (pq)U$
        6. $(p+q)U = pU + qU$
        7. $p(U+V) = pU + pV$
        8. $1U = U$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The most familar example of vector spaces are the collections of single column arrays that we have been referring to as "vectors" throughout the previous chapter.  The name given to the collection of all $n\times 1$ arrays is known as Euclidean $n$-space, and is given the symbol $\mathbb{R}^n$.  The required definitions of addition and scalar multiplication in $\mathbb{R}^n$ are those described for matrices in [Matrix Algebra](Matrix_Algebra.ipynb).  We will leave it to the reader to verify that these operations satisfy the list of requirements listed above.  


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The algebra of vectors can be visualized by interpreting the vectors as arrows.  This is easiest to see with an example in $\mathbb{R}^2$.  

        $$
        \begin{equation}
        U_1 = \left[ \begin{array}{r} 1 \\ 3  \end{array}\right] \hspace{1cm} 
        U_2 = \left[ \begin{array}{r} 2 \\  -1   \end{array}\right]
        \end{equation}
        $$

        The vector $U_1$ can be visualized as an arrow that points in the direction defined by 1 unit to the right, and 3 units up.
        """
    )
    return


@app.cell
def _():
    # Cell tags: hide-input
    # '%matplotlib inline\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\noptions = {"head_width":0.1, "head_length":0.2, "length_includes_head":True}\n\nax.arrow(0,0,1,3,fc=\'b\',ec=\'b\',**options)\n\nax.text(1,2,\'$U_1$\')\n\nax.set_xlim(0,5)\nax.set_ylim(0,5)\nax.set_aspect(\'equal\')\nax.grid(True,ls=\':\')' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It is important to understand that it is the *length and direction* of this arrow that defines $U_1$, not the actual position.  We could draw the arrow in any number of locations, and it would still represent $U_1$.
        """
    )
    return


@app.cell
def _(options, plt):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _ax.arrow(0, 0, 1, 3, fc='b', ec='b', **options)
    _ax.arrow(3, 0, 1, 3, fc='b', ec='b', **options)
    _ax.arrow(0, 2, 1, 3, fc='b', ec='b', **options)
    _ax.arrow(2, 1, 1, 3, fc='b', ec='b', **options)
    _ax.set_xlim(0, 5)
    _ax.set_ylim(0, 5)
    _ax.set_aspect('equal')
    _ax.grid(True, ls=':')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When we perform a scalar multiplication, such as $2U_1$, we interpret it as multiplying the *length of the arrow* by the scalar.
        """
    )
    return


@app.cell
def _(np, options, plt):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _ax.arrow(0, 0, 1, 3, fc='b', ec='b', **options)
    _ax.arrow(2, 0, 2, 6, fc='r', ec='r', **options)
    _ax.text(1, 2, '$U_1$')
    _ax.text(4, 5, '$2U_1$')
    _ax.set_xlim(0, 6)
    _ax.set_ylim(0, 6)
    _ax.set_aspect('equal')
    _ax.grid(True, ls=':')
    _ax.set_xticks(np.arange(0, 7, step=1))
    _ax.set_yticks(np.arange(0, 7, step=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If the scalar is negative, we interpret the scalar multiplication as *reversing the direction* of the arrow, as well as changing the length.
        """
    )
    return


@app.cell
def _(np, options, plt):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _ax.arrow(0, 0, 1, 3, fc='b', ec='b', **options)
    _ax.arrow(4, 6, -2, -6, fc='r', ec='r', **options)
    _ax.text(1, 2, '$U_1$')
    _ax.text(3, 1, '$-2U_1$')
    _ax.set_xlim(0, 6)
    _ax.set_ylim(0, 6)
    _ax.set_aspect('equal')
    _ax.grid(True, ls=':')
    _ax.set_xticks(np.arange(0, 7, step=1))
    _ax.set_yticks(np.arange(0, 7, step=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can interpret the sum of two vectors as the result of aligning the two arrows tip to tail.
        """
    )
    return


@app.cell
def _(np, options, plt):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _ax.arrow(0, 0, 1, 3, fc='b', ec='b', **options)
    _ax.arrow(1, 3, 2, -1, fc='b', ec='b', **options)
    _ax.arrow(0, 0, 3, 2, fc='r', ec='r', **options)
    _ax.text(1, 2, '$U_1$')
    _ax.text(2, 3, '$U_2$')
    _ax.text(2, 1, '$U_1+U_2$')
    _ax.set_xlim(0, 4)
    _ax.set_ylim(0, 4)
    _ax.set_aspect('equal')
    _ax.grid(True, ls=':')
    _ax.set_xticks(np.arange(0, 5, step=1))
    _ax.set_yticks(np.arange(0, 5, step=1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{equation}
        \left[ \begin{array}{r} 1 \\ 3 \end{array}\right] 
        + \left[ \begin{array}{r} 2 \\ -1  \end{array}\right]= 
        \left[ \begin{array}{r} 3 \\ 2\end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are many other examples of vector spaces, but we will wait to provide these until after we have discussed more of the fundamental vector space concepts using $\mathbb{R}^n$ as the setting.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
