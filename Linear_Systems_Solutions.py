import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear Systems
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
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Consider the system of equations given below. Find a value for the coefficient $a$ so that the system has exactly one solution. For that value of $a$, find the solution to the system of equations and plot the lines formed by your equations to check your answer.

        $$
        \begin{eqnarray*}
        ax_1 + 2x_2 & = & 1 \\
        4x_1 + 8x_2 & = & 12 \\
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If $a=1$ the two lines are parallel and do not intersect.  For any other value of $a$ the lines intersect at exactly one point.  For example, we take $a = -1$. By plotting the two lines formed by these equations and examining where they intersect we can see that the solution to this system is $x_1 = 1,$ $x_2 = 1$
        """
    )
    return


@app.cell
def _(np, plt):
    x=np.linspace(-5,5,100)

    fig, ax = plt.subplots()
    ax.plot(x,(x+1)/2)
    ax.plot(x,(12-4*x)/8)

    ax.text(-2.9,3.1,'$4x_1 + 8x_2 = 12$')
    ax.text(-3,-1.6,'$-x_1 + 2x_2 = 1$')

    ax.set_xlim(-4,4)
    ax.set_ylim(-2,6)
    ax.axvline(color='k',linewidth = 1)
    ax.axhline(color='k',linewidth = 1)

    ax.set_xticks(np.linspace(-4,4,9))
    ax.set_yticks(np.linspace(-2,6,9))
    ax.set_aspect('equal')
    ax.grid(True,ls=':')
    return ax, fig, x


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Find a value for the coefficient $a$ so that the system has no solution. Plot the lines formed by your equations to verify there is no solution.

        $$
        \begin{eqnarray*}
         ax_1 + 2x_2 & = & 1 \\
        4x_1 + 8x_2 & = & 12 \\
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If $a = 1$ then the lines are parallel and there is no solution to the system.
        """
    )
    return


@app.cell
def _(np, plt):
    x_5 = np.linspace(-5, 5, 100)
    fig_1, ax_1 = plt.subplots()
    ax_1.plot(x_5, (1 - x_5) / 2)
    ax_1.plot(x_5, (12 - 4 * x_5) / 8)
    ax_1.text(-2.8, 3.1, '$4x_1 + 8x_2 = 12$')
    ax_1.text(0.4, -1.4, '$x_1 + 2x_2 = 1$')
    ax_1.set_xlim(-4, 4)
    ax_1.set_ylim(-2, 6)
    ax_1.axvline(color='k', linewidth=1)
    ax_1.axhline(color='k', linewidth=1)
    ax_1.set_xticks(np.linspace(-4, 4, 9))
    ax_1.set_yticks(np.linspace(-2, 6, 9))
    ax_1.set_aspect('equal')
    ax_1.grid(True, ls=':')
    return ax_1, fig_1, x_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Consider the system of equations given below. Under what conditions on the coefficients $a,b,c,d,e,f$ does this system have only one solution? No solutions? An infinite number of solutions? (*Think about what must be true of the slopes and intercepts of the lines in each case.*)


        $$
        \begin{eqnarray*}
        ax_1 + bx_2 & = & c \\
        dx_1 + ex_2 & = & f \\
        \end{eqnarray*}
        $$

            



        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Using algebra, we can convert the arbitrary equation $ax_1 + bx_2 = c$ to its slope-intercept form $x_2 = - \frac{a}{b}x_1 + \frac{c}{b}$. Two lines intersect only once in the plane if and only if their slopes are different.  Therefore if $\frac{a}{b} \neq \frac{d}{e}$ then the system has exactly one solution. In the case that the slopes are equal, we examine the y-intercepts of the equations to determine whether there exists no solutions or an infinite number of solutions. If $\frac{c}{b} = \frac{f}{e}$ then the y-intercepts are equal and the two equations describe the same line.  In this case there are an infinite number of solutions. On the other hand, if $\frac{c}{b} \neq \frac{f}{e}$ then the lines are parallel and there exists no solutions. In summary:

        - If $\frac{a}{b} \neq \frac{d}{e}$ then there exists exactly one solution
        - If $\frac{a}{b} = \frac{d}{e}$ and $\frac{c}{b} = \frac{f}{e}$ then there are an infinite number of solutions
        - If $\frac{a}{b} = \frac{d}{e}$ and $\frac{c}{b} \neq \frac{f}{e}$ then there are no solutions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Gaussian Elimination
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** In the system below, use a row operation functions to produce a zero coefficient in the location of the coefficient 12.  Do this first by hand and then create a NumPy array to represent the system and use the row operation functions.  (*There are two ways to create the zero using $\texttt{RowAdd}$.  Try to find both.*)

        $$
        \begin{eqnarray*}
        4x_1 - 2x_2 + 7x_3 & = & 2\\
        x_1 + 3x_2 + 12x_3 & = & 4\\
        -7x_1 \quad\quad - 3x_3 & = & -1 
        \end{eqnarray*}
        $$

        """
    )
    return


@app.cell
def _(lag, np):
    C = np.array([[4,-2,7,2],[1,3,12,4],[-7,0,-3,-1]])
    C1 = lag.RowAdd(C,0,1,-12/7)
    C2 = lag.RowAdd(C,2,1,4)
    print(C,'\n')
    print(C1,'\n')
    print(C2,'\n')
    return C, C1, C2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Create a NumPy array to represent the system below.  Determine which coefficient should be zero in order for the system to be upper triangular.  Use $\texttt{RowAdd}$ to carry out the row operation and then print your results.
          
        $$
        \begin{eqnarray*}
        3x_1 + 4x_2 \, - \,\,\,\,\,  x_3 &   =   & -6\\
        -2x_2   +  10x_3  &   =   & -8\\
        4x_2   \,  - \,\, 2x_3 &  =  & -2 
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    A = np.array([[3,4,-1,-6],[0,-2,10,-8],[0,4,-2,-2]])

    # B is obtained after performing RowAdd on A.
    B = lag.RowAdd(A, 1, 2, 2)
    print(B)
    return A, B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Carry out the elimination process on the following system.  Define a NumPy array and make use of the row operation functions.  Print the results of each step.  Write down the upper triangular system represented by the array after all steps are completed.

        $$
        \begin{eqnarray*}
        x_1 - x_2 + x_3 & = & 3\\
        2x_1 + x_2 + 8x_3 & = & 18\\
        4x_1 + 2x_2 -3x_3 & = & -2 
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    A_1 = np.array([[1, -1, 1, 3], [2, 1, 8, 18], [4, 2, -3, -2]])
    A1 = lag.RowScale(A_1, 0, 1.0 / A_1[0][0])
    A2 = lag.RowAdd(A1, 0, 1, -A_1[1][0])
    A3 = lag.RowAdd(A2, 0, 2, -A2[2][0])
    A4 = lag.RowScale(A3, 1, 1.0 / A3[1][1])
    A5 = lag.RowAdd(A4, 1, 2, -A4[2][1])
    A6 = lag.RowScale(A5, 2, 1.0 / A5[2][2])
    print(A_1, '\n')
    print(A1, '\n')
    print(A2, '\n')
    print(A3, '\n')
    print(A4, '\n')
    print(A5, '\n')
    print(A6, '\n')
    return A1, A2, A3, A4, A5, A6, A_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Use row operations on the system below to produce an **lower triangular** system.  The first equation of the lower triangular system should contain only $x_1$ and the second equation should contain only $x_1$ and $x_2$.

          $$
        \begin{eqnarray*}
        x_1 + 2x_2 + x_3 & = & 3\\
        3x_1 - x_2 - 3x_3 & = & -1\\
        2x_1 + 3x_2 + x_3 & = & 4
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    A_2 = np.array([[1, 2, 1, 3], [3, -1, -3, -1], [2, 3, 1, 4]])
    print('A: \n', A_2, '\n')
    B_1 = lag.RowAdd(A_2, 2, 0, -1)
    C_1 = lag.RowAdd(B_1, 2, 1, 3)
    L = lag.RowAdd(C_1, 1, 0, 1 / 8)
    print('L: \n', L, '\n')
    return A_2, B_1, C_1, L


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The corresponding lower triangular system is

        $$
        \begin{eqnarray*}
        0.125x_1 \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, & = & 0.375\\
        9x_1 + 8x_2 \,\,\,\,\,\,\,\,\,\,\,\,\,\,\  & = & 11\\
        2x_1 + 3x_2 + 1x_3 & = & 4 
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Use elimination to determine whether the following system is **consistent** or **inconsistent**.
         
        $$
        \begin{eqnarray*}
        x_1 - x_2 - x_3 & = & 4\\
        2x_1 - 2x_2 - 2x_3 & = & 8\\
        5x_1 - 5x_2 - 5x_3 & = & 20 
        \end{eqnarray*}
        $$


        """
    )
    return


@app.cell
def _(lag, np):
    A_3 = np.array([[1, -1, -1, 4], [2, -2, -2, 8], [5, -5, -5, 20]])
    B1 = lag.RowAdd(A_3, 0, 1, -2)
    B2 = lag.RowAdd(B1, 0, 2, -5)
    print(B1)
    print(B2)
    return A_3, B1, B2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this system the second equation and third equation are just multiples of the first.  Any solution to one of the equations is a solution to all three equations.  We can get a solution to the first equation by choosing values for $x_2$ and $x_3$ and then calculating the value of $x_1$ that satisfies the equation.  One such solution is $x_1 = 6$, $x_2=1$, $x_3=1$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Use elimination to show that this system of equations has no solution.

          $$
        \begin{eqnarray*}
        x_1  +  x_2 +  x_3 & = & 0\\
        x_1 -  x_2 + 3x_3 & = & 3\\
        -x_1 - x_2 - x_3 & = & 2 
        \end{eqnarray*}
        $$
          

        """
    )
    return


@app.cell
def _(lag, np):
    A_4 = np.array([[1, 1, 1, 0], [1, -1, 3, 3], [-1, -1, -1, 2]])
    B1_1 = lag.RowAdd(A_4, 0, 1, -1)
    print(B1_1)
    B2_1 = lag.RowAdd(B1_1, 0, 2, 1)
    print(B2_1)
    return A_4, B1_1, B2_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The third row of $\texttt{B2}$ indicates that the system is inconsistent.  This row represents the equation $0x_3 = 2$, which cannot be true for any value of $x_3$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Use  $\texttt{random}$ module to produce a $3\times 4$ array which contains random integers between $0$ and $5$. Write code that performs a row operation that produces a zero in the first row, third column.  Run the code several times to be sure that it works on **different** random arrays.  Will the code ever fail?
        """
    )
    return


@app.cell
def _(lag, np):
    A_5 = np.random.randint(0, 5, size=(3, 4))
    print('A: \n', A_5, '\n')
    if A_5[1, 2] == 0:
        A_5 = lag.RowSwap(A_5, 0, 1)
    else:
        A_5 = lag.RowAdd(A_5, 1, 0, -A_5[0, 2] / A_5[1, 2])
    print('A after one row operation: \n', A_5, '\n')
    return (A_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The code will not fail to produce a matrix with a zero in the specified position.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Starting from the array that represents the upper triangular system in **Example 1** ($\texttt{A5}$), use the row operations to produce an array of the following form.  Do one step at a time and again print your results to check that you are successful.  

        $$
        \begin{equation}
        \left[ \begin{array}{cccc} 1 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 2 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    B_2 = np.array([[1, -1, 1, 3], [0, 1, 2, 4], [0, 0, 1, 2]])
    B1_2 = lag.RowAdd(B_2, 1, 0, 1)
    B2_2 = lag.RowAdd(B1_2, 2, 0, -3)
    B3 = lag.RowAdd(B2_2, 2, 1, -2)
    print(B_2, '\n')
    print(B1_2, '\n')
    print(B2_2, '\n')
    print(B3, '\n')
    return B1_2, B2_2, B3, B_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 9:** Redo **Example 2** using the $\texttt{random}$ module to produce a $3\times 4$ array made up of random floats instead of random integers.  
        """
    )
    return


@app.cell
def _(lag, np):
    D = np.random.rand(3,4)
    D1 = lag.RowScale(D,0,1.0/D[0][0])
    D2 = lag.RowAdd(D1,0,1,-D[1][0])
    D3 = lag.RowAdd(D2,0,2,-D2[2][0])
    D4 = lag.RowScale(D3,1,1.0/D3[1][1])
    D5 = lag.RowAdd(D4,1,2,-D4[2][1])
    D6 = lag.RowScale(D5,2,1.0/D5[2][2])
    print(D,'\n')
    print(D1,'\n')
    print(D2,'\n')
    print(D3,'\n')
    print(D4,'\n')
    print(D5,'\n')
    print(D6,'\n')
    return D, D1, D2, D3, D4, D5, D6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 10:** Write a loop that will execute the elimination code in **Example 2** on 1000 different $3\times 4$ arrays of random floats to see how frequently it fails.  
        """
    )
    return


@app.cell
def _(lag, np):
    for count in range(1000):
        E = np.random.rand(3,4)
        E1 = lag.RowScale(E,0,1.0/E[0][0])
        E2 = lag.RowAdd(E1,0,1,-E[1][0])
        E3 = lag.RowAdd(E2,0,2,-E2[2][0])
        E4 = lag.RowScale(E3,1,1.0/E3[1][1])
        E5 = lag.RowAdd(E4,1,2,-E4[2][1])
        E6 = lag.RowScale(E5,2,1.0/E5[2][2])
    return E, E1, E2, E3, E4, E5, E6, count


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Matrix Algebra

        **Exercise 1:** Calculate $-2E$, $G+F$, $4F-G$, $HG$, and $GE$ using the following matrix definitions.  Do the exercise on paper first, then check by doing the calculation with NumPy arrays.

        $$
        \begin{equation}
        E = \left[ \begin{array}{r} 5 \\ -2 \end{array}\right] \hspace{1cm} 
        F = \left[ \begin{array}{rr} 1 & 6 \\ 2 & 0 \\ -1 & -1 \end{array}\right] \hspace{1cm}
        G = \left[ \begin{array}{rr} 2 & 0\\ -1 & 3 \\ -1 & 6 \end{array}\right] \hspace{1cm}
        H = \left[ \begin{array}{rrr} 3 & 0 & 1 \\ 1 & -2 & 2 \\ 3 & 4 & -1\end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(np):
    E_1 = np.array([[5], [-2]])
    F = np.array([[1, 6], [2, 0], [-1, -1]])
    G = np.array([[2, 0], [-1, 3], [-1, 6]])
    H = np.array([[3, 0, 1], [1, -2, 2], [3, 4, -1]])
    print(-2 * E_1, '\n')
    print(G + F, '\n')
    print(4 * F - G, '\n')
    print(H @ G, '\n')
    print(G @ E_1)
    return E_1, F, G, H


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Find the values of $x$ and $y$ so that this equation holds.

        $$
        \begin{equation}
        \left[ \begin{array}{rr} 1 & 3 \\ -4 & 2 \end{array}\right]
        \left[ \begin{array}{rr} 3 & x \\ 2 & y \end{array}\right]=
        \left[ \begin{array}{rr} 9 & 10 \\ -8 & 16 \end{array}\right] 
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Recall that matrix multiplication can be carried out column by column.  Given a matrix equation of the form $AB = C$, multiplying $A$ by the $j^{\text{th}}$ column of $B$ gives the $j^{\text{th}}$ column of $C$. This view of matrix multiplication gives the following linear system which we can solve with elimination

        $$
        \begin{equation}
        \left[ \begin{array}{rr} 1 & 3 \\ -4 & 2 \end{array}\right]
        \left[ \begin{array}{rr} x \\ y \end{array}\right] =
        \left[ \begin{array}{rr} x + 3y \\ -4x + 2y \end{array}\right] =
        \left[ \begin{array}{cccc} 10\\ 16 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    A_6 = np.array([[1, 3, 10], [-4, 2, 16]])
    A1_1 = lag.RowAdd(A_6, 0, 1, 4)
    A2_1 = lag.RowScale(A1_1, 1, 1 / 14)
    A3_1 = lag.RowAdd(A2_1, 1, 0, -3)
    print(A_6, '\n')
    print(A1_1, '\n')
    print(A2_1, '\n')
    print(A3_1)
    return A1_1, A2_1, A3_1, A_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Therefore $x = -2$ and $y = 4$.

        Alternatively, if we carry out the matrix multiplication completely, we arrive at the same system to be solved.

        $$
        \begin{equation}
        \left[ \begin{array}{rr} 9 & x + 3y \\ -8 & -4x + 2y \end{array}\right] =
        \left[ \begin{array}{cccc} 9 & 10\\ -8 & 16 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Define NumPy arrays for the matrices $H$ and $G$ given below.  

        $$
        \begin{equation}
        H = \left[ \begin{array}{rrr} 3 & 3 & -1  \\ -3 & 0 & 8 \\  1 & 6 & 5 \end{array}\right]\hspace{2cm}
        G = \left[ \begin{array}{rrrr} 1 & 5 & 2 & -3 \\ 7 & -2 & -3 & 0 \\ 2 & 2 & 4 & 6\end{array}\right]
        \end{equation}
        $$

        $(a)$ Multiply the second and third column of $H$ with the first and second row of $G$.  Use slicing to make subarrays.  Does the result have any relationship to the full product $HG$?

        $(b)$ Multiply the first and second row of $H$ with the second and third column of $G$.  Does this result have any relationship to the full product $HG$?

        """
    )
    return


@app.cell
def _(np):
    H_1 = np.array([[3, 3, -1], [-3, 0, 8], [1, 6, 5]])
    G_1 = np.array([[1, 5, 2, -3], [7, -2, -3, 0], [2, 2, 4, 6]])
    print(H_1 @ G_1, '\n')
    print(H_1[:, 1:3] @ G_1[0:2, :], '\n')
    print(H_1[0:2, :] @ G_1[:, 1:3])
    return G_1, H_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The second product is a $2 \times 2$ matrix that is a **submatrix** of the larger matrix $HG$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Generate a $4\times 4$ matrix $B$ with random integer entries.  Compute matrices $P = \frac12(B+B^T)$ and $Q = \frac12(B-B^T)$.  Rerun your code several times to get different matrices.  What do you notice about $P$ and $Q$?  Explain why it must always be true.       
        """
    )
    return


@app.cell
def _(np):
    B_3 = np.random.randint(size=(4, 4), high=10, low=-10)
    P = 1 / 2 * (B_3 + B_3.transpose())
    Q = 1 / 2 * (B_3 - B_3.transpose())
    print(B_3, '\n')
    print(P, '\n')
    print(Q)
    return B_3, P, Q


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $P$ is always a symmetric matrix and $Q$ is always a skew-symmetric matrix.
            
        To see why $P$ is always symmetric, recall that $P$ being symmetric is equivalent to saying $p_{ij} = p_{ji}$ for all entries $p_{ij}$ in $P$. Now notice that 

        $$
        \begin{equation}   
        p_{ij} = \frac{1}{2}(b_{ij} + b_{ji}) = \frac{1}{2}(b_{ji} + b_{ij}) = p_{ji}
        \end{equation}
        $$

        and therefore $P$ is symmetric.
           
        To see why $Q$ is always skew-symmetric, recall that $Q$ being skew-symmetric is equivalent to saying $q_{ij} = -q_{ji}$ for all entries $q_{ij}$ in $Q$. Now notice that
           
        $$
        \begin{equation}
        q_{ij} = \frac{1}{2}(b_{ij} - b_{ji}) = -\frac{1}{2}(b_{ji} - b_{ij}) = -q_{ji}
        \end{equation}
        $$

        and therefore $Q$ is skew-symmetric.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Just as the elements of a matrix can be either integers or real numbers, we can also have a matrix with elements that are themselves matrices. A **block matrix** is any matrix that we have interpreted as being partitioned into these **submatrices**. Evaluate the product $HG$ treating the $H_i$'s and $G_j$'s as the elements of their respective matrices. If we have two matrices that can be multiplied together normally, does any partition allow us to multiply using the submatrices as the elements?

        $$
        \begin{equation}
        H = \left[ \begin{array}{cc|cc} 1 & 3 & 2 & 0  \\ -1 & 0 & 3 & 3 \\ \hline 2 & 2 & -2 & 1 \\ 0 & 1 & 1 & 4 \end{array}\right] = \left[ \begin{array}{} H_1 & H_2 \\ H_3 & H_4\end{array} \right] \hspace{2cm} 
        G = \left[ \begin{array}{cc|c} 3 & 0 & 5 \\ 1 & 1 & -3 \\ \hline 2 & 0 & 1 \\ 0 & 2 & 1\end{array}\right] = \left[ \begin{array}{} G_1 & G_2 \\ G_3 & G_4\end{array} \right]
        \end{equation}
        $$
        """
    )
    return


app._unparsable_cell(
    r"""
    H = np.array([[1,3,2,0],[-1,0,3,3],[2,2,-2,1],[0,1,1,4]])
    G = np.array([[3,0,5],[1,1,-3],[2,0,1],[0,2,1]])b
    H1 = np.array([[1,3],[-1,0]])
    H2 = np.array([[2,0],[3,3]])
    H3 = np.array([[2,2],[0,1]])
    H4 = np.array([[-2,1],[1,4]])
    G1 = np.array([[3,0],[1,1]])
    G2 = np.array([[5],[-3]])
    G3 = np.array([[2,0],[0,2]])
    G4 = np.array([[1],[1]])
    HG1 = H1@G1 + H2@G3
    HG2 = H1@G2 + H2@G4
    HG3 = H3@G1 + H4@G3
    HG4 = H3@G2 + H4@G4
    print(H@G,'\n')
    print(HG1,'\n')
    print(HG2,'\n')
    print(HG3,'\n')
    print(HG4)
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Not every partition of a matrix will let us multiply them in this way. If we had instead partitioned $H$ like this:

        $$
        \begin{equation}
        H = \left[ \begin{array}{c|ccc} 1 & 3 & 2 & 0  \\ -1 & 0 & 3 & 3 \\ \hline 2 & 2 & -2 & 1 \\ 0 & 1 & 1 & 4 \end{array}\right] = \left[ \begin{array}{} H_1 & H_2 \\ H_3 & H_4\end{array} \right] \hspace{2cm} 
        \end{equation}
        $$

        then many of the products calculated above (such as $H_1G_1$) would not be defined.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Inverse Matrices

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Solve the following system of equations using an inverse matrix.

        $$
        \begin{eqnarray*}
        2x_1 + 3x_2 + x_3 & = & 4\\
        3x_1 + 3x_2 + x_3 & = & 8\\
        2x_1 + 4x_2 + x_3 & = & 5 
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can write the matrix equation $AX = B$ for the given system of equation which looks like:

        $$
        AX=
        \left[ \begin{array}{rrrr} 2 & 3 & 1  \\ 3 & 3 & 1  \\ 2 & 4 & 1   \end{array}\right]
        \left[ \begin{array}{rrrr} x_1 \\ x_2 \\ x_3 \end{array}\right]=
        \left[ \begin{array}{rrrr} 4 \\ 8 \\ 5  \end{array}\right]=
        B
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    A_7 = np.array([[2, 3, 1], [3, 3, 1], [2, 4, 1]])
    print('A: \n', A_7, '\n')
    A_inv = lag.Inverse(A_7)
    b = np.array([[4], [8], [5]])
    print('b: \n', b, '\n')
    print('A_inv: \n', A_inv, '\n')
    x_6 = A_inv @ b
    print('x: \n', x_6, '\n')
    return A_7, A_inv, b, x_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Let $A$ and $B$ be two random $4\times 4$ matrices.  Demonstrate using Python that $(AB)^{-1}=B^{-1}A^{-1}$ for the matrices.
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
        **Exercise 3:** Explain why $(AB)^{-1}=B^{-1}A^{-1}$ by using the definition given in this section.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:**  Solve the system $AX=B$ by finding $A^{-1}$ and computing $X=A^{-1}B$.

        $$
        A = \left[ \begin{array}{rrrr} 1 & 2 & -3 \\ -1 & 1 & -1  \\ 0 & -2 & 3  \end{array}\right] \quad\quad
        B = \left[ \begin{array}{rrrr} 1  \\ 1 \\ 1  \end{array}\right]
        $$    
        """
    )
    return


@app.cell
def _(lag, np):
    A_8 = np.array([[1, 2, -3], [-1, 1, -1], [0, -2, 3]])
    print('A: \n', A_8, '\n')
    A_inverse = lag.Inverse(A_8)
    B_4 = np.array([[1], [1], [1]])
    print('A inverse is : \n', A_inverse, '\n')
    X = A_inverse @ B_4
    print('X: \n', X, '\n')
    return A_8, A_inverse, B_4, X


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Find a $3 \times 3 $ matrix $Y$ such that $AY = C$.

        $$
        A = \left[ \begin{array}{rrrr} 3 & 1 & 0 \\ 5 & 2 & 1 \\ 0 & 2 & 3\end{array}\right]\hspace{2cm}
        C = \left[ \begin{array}{rrrr} 1 & 2 & 1 \\ 3 & 4 & 0 \\ 1 & 0 & 2 \end{array}\right]\hspace{2cm}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    A_9 = np.array([[3, 1, 0], [5, 2, 1], [0, 2, 3]])
    C_2 = np.array([[1, 2, 1], [3, 4, 0], [1, 0, 2]])
    print('A: \n', A_9, '\n')
    print('C: \n', C_2, '\n')
    A_inv_1 = lag.Inverse(A_9)
    Y = A_inv_1 @ C_2
    print('A inverse is : \n', A_inv_1, '\n')
    print('Y: \n', Y, '\n')
    return A_9, A_inv_1, C_2, Y


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Let $A$ be a random $4 \times 1$ matrix and $B$ is a random $ 1 \times 4 $ matrix. Use Python to demonstrate that the product $ AB $ is not invertible. Do you expect this to be true for any two matrices $P$ and $Q$ such that $P$ is an $ n \times 1 $ matrix and $Q$ is a $ 1 \times n$ matrix ? Explain.
        """
    )
    return


@app.cell
def _(lag, np):
    A_10 = np.random.randint(0, 9, size=(4, 1))
    B_5 = np.random.randint(0, 9, size=(1, 4))
    print('A: \n', A_10, '\n')
    print('B: \n', B_5, '\n')
    AB = A_10 @ B_5
    AB = lag.FullRowReduction(AB)
    print('AB: \n', AB, '\n')
    return AB, A_10, B_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It can be clearly seen from the above code cell that the reduced form of the product $AB$ contains zero enteries on the diagonal. This shows that $AB$ is not invertible. It is expected to be true for any two general matrices $P$ and $Q$ where $P$ is an $ n \times 1 $ matrix and $Q$ is a $ 1 \times n$ matrix. 

        Let $$
        P = \left[ \begin{array}{rrrr} a \\ b \\ c \\  d\\..\\.. \end{array}\right]
        $$
        and $$
        Q = \left[ \begin{array}{rrrr} w & x &y&z&.&.&. \end{array}\right]
        $$

        $$
        PQ = \left[ \begin{array}{rrrr} aw & ax &ay& az &.&.&. \\ bw & bx & by & bz &.&.&.\\ cw & cx & cy & cz&.&.&. \\ dw & dx & dy & dz&.&.&. \\.&.&.&.&.&.&.\\. &.&.&.&.&.&.\end{array}\right]
        $$
         
        If we perform a $\texttt{RowAdd}$ to create a zero at any position in a row using another row, the entire row becomes zero. Notice that each row is just a multiple of the first row.  This means that when we carry out row reduction on $PQ$, the resulting **upper triangular** matrix have zeros in all entries except for the first row.  This implies that the product $PQ$ is non-invertible.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Let $A$ be a random $ 3 \times 3$ matrices. Demonstrate using Python that $(A^T)^{-1} = (A^{-1})^T$ for the matrix. Use this property to explain why $A^{-1}$ must be symmetric if $A$ is symmetric.
        """
    )
    return


@app.cell
def _(lag, np):
    A_11 = np.random.randint(0, 9, size=(3, 3))
    print('A: \n', A_11, '\n')
    A_inv_2 = lag.Inverse(A_11)
    A_inv_trans = A_inv_2.transpose()
    print('A_inv_transpose: \n', np.round(A_inv_trans, 8), '\n')
    A_trans = A_11.transpose()
    A_trans_inv = lag.Inverse(A_trans)
    print('A_trans_inverse: \n', np.round(A_trans_inv, 8), '\n')
    if np.array_equal(A_trans_inv, A_inv_trans):
        print('A_inverse_transpose and A_transpose_inverse are equal. ')
    else:
        print('A_inverse_transpose and A_transpose_inverse are not equal. ')
    return A_11, A_inv_2, A_inv_trans, A_trans, A_trans_inv


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From the computations done in the code cell above, we verified that $ (A^{-1})^T = (A^T)^{-1} $.  If $A$ is symmetric, then $A^T = A$ and we can replace $A^T$ with $A$ on the right side of the equation to get $(A^{-1})^T = A^{-1} $, which means that $A^{-1}$ is also symmetric.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Consider the following $ 4 \times 4 $ matrix:

        $$
        A = \left[ \begin{array}{rrrr} 4 & x_1 & 0 & 0 \\ 0 & x_2 & 0 & 0 \\ 0 & x_3 & 1 & 0 \\ 0 & x_4 & 0 & 3 \end{array}\right]
        $$

          $(a)$ Find the condition on $x_1$, $x_2$, $x_3$ or $x_4$ for which $A^{-1}$ exists. Assuming that condition is true, find the inverse of $A$.

          $(b)$ Use Python to check if $ A^{-1}A = I $ when $x_1 = 4$, $x_2 = 1$, $x_3 = 1$, and $x_4 = 3$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Solution: 
        (a) After carrying out row operations on $A$, we get the following upper triangular matrix.

        $$
        \left[ \begin{array}{rrrr} 4 & x_1 & 0 & 0 \\ 0 & x_2 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 3 \end{array}\right]
        $$

        $A$ will be invertible if and only if all the diagonal enteries are non-zero, so $A$ will be invertible if and only if $x_2\neq 0$.  In this case, $[A|I]$ is

        $$
        \begin{equation}
        [A|I] = \left[ \begin{array}{rrrr|rrrr} 
        4 & x_1 & 0 & 0 & 1 & 0 & 0 & 0 \\ 
        0 & x_2 & 0 & 0 & 0 & 1 & 0 & 0 \\
        0 & x_3 & 1 & 0 & 0 & 0 & 1 & 0 \\
         0& x_4 & 0 & 3 & 0 & 0 & 0 & 1 \\ 
        \end{array}\right]
        \end{equation}
        $$

        row operations give $A^{-1}$ as follows.

        $$
        A^{-1} = \left[ \begin{array}{rrrr} 1/4 & -x_1/4x_2 & 0 & 0 \\ 0 & x_2 & 0 & 0 \\ 0 & -x_3/x_2 & 1 & 0 \\ 0 & -x_4/3x_2 & 0 & 1/3 \end{array}\right]
        $$

        """
    )
    return


@app.cell
def _(np):
    x_1 = 1
    x_2 = 2
    x_3 = 4
    x_4 = 0
    A_12 = np.array([[4, x_1, 0, 0], [0, x_2, 0, 0], [0, x_3, 1, 0], [0, x_4, 0, 3]])
    A_inv_3 = np.array([[1 / 4, -x_1 / 4 / x_2, 0, 0], [0, x_2, 0, 0], [0, -x_3 / x_2, 1, 0], [0, -x_4 / 3 / x_2, 0, 1 / 3]])
    print('A@A_inv: \n', A_12 @ A_inv_3, '\n')
    return A_12, A_inv_3, x_1, x_2, x_3, x_4


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 9:** Apply the methods used in this section to compute a right inverse of the matrix $A$.

        $$
        A = \left[ \begin{array}{rrrr} 1 & 0 & 0 & 2 \\ 0 & -1 & 1 & 4 \end{array}\right]
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        Let $C$ be the right inverse of $A$, then $AC= I$, and $C$ is a $ 4 \times 2 $ matrix.

        $$
        AC=
        \left[ \begin{array}{rrrr} 1 & 0 & 0 & 2 \\ 0 & -1 & 1 & 4  \end{array}\right]
        \left[ \begin{array}{rrrr} x_1 & y_1  \\ x_2 & y_2  \\ x_3 & y_3 \\ x_4 & y_4   \end{array}\right]=
        \left[ \begin{array}{rrrr} 1 & 0  \\ 0 & 1  \end{array}\right]=
        I
        $$

        If we understand the product $AC$ in terms of the columns of $C$, we get two linear systems to solve for $x_i$ and $y_i$ which are as follows:

        $$
        \left[ \begin{array}{rrrr} 1 & 0 & 0 & 2 \\ 0 & -1 & 1 & 4  \end{array}\right]
        \left[ \begin{array}{r}  x_1 \\  x_2  \\ x_3 \\ x_4  \end{array}\right]=
        \left[ \begin{array}{r}  1 \\  0   \end{array}\right]
        $$

                                                                       

        $$
        \left[ \begin{array}{rrrr} 1 & 0 & 0 & 2 \\ 0 & -1 & 1 & 4  \end{array}\right]
        \left[ \begin{array}{r}  y_1 \\  y_2  \\ y_3 \\ y_4  \end{array}\right]=
        \left[ \begin{array}{r}  0 \\ 1 \end{array}\right]
        $$


        It can be clearly seen that $x_3, x_4, y_3$, and $y_4$ are free variables due to which there are infinite number of matrices $C$ which can be the right inverse of $A$. In other words, $C$ is not unique. One possibility is: 

        $$
        C = \left[ \begin{array}{rrrr} 1 & 0 \\ 0 & -1 \\ 0 & 0 \\ 0 & 0 \end{array}\right]
        $$

        We check that $ AC = I $ in the code cell below.
        """
    )
    return


@app.cell
def _(np):
    A_13 = np.array([[1, 0, 0, 2], [0, -1, 1, 4]])
    C_3 = np.array([[1, 0], [0, -1], [0, 0], [0, 0]])
    print('AC = \n', A_13 @ C_3, '\n')
    return A_13, C_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### LU Factorization

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Solve $ AX = B $ using $ A = LU $ and the $L$, $U$, and $B$ given below.  Compute $LUX$ to verify your answer. 

        $$
        \begin{equation}
        A = \left[ \begin{array}{ccc} 5 & 2 & 1 \\ 5 & 3 & 0 \\ -5 & -2 & -4  \end{array}\right] \hspace{2cm} 
        B = \left[ \begin{array}{c} 4 \\ 7 \\ 8  \end{array}\right] \hspace{2cm} 
        L = \left[ \begin{array}{ccc} 1 & 0 & 0 \\ 1 & 1 & 0  \\ -1 & 0 & 1  \end{array}\right] \hspace{2cm} 
        U = \left[ \begin{array}{ccc} 5 & 2 & 1 \\ 0 & 1 & -1  \\ 0 & 0 & 3  \end{array}\right] 
        \end{equation}
        $$
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
        **Exercise 2:** Solve $ AX = B $ using $ A = LU $ and the $L$, $U$, and $B$ given below.  Compute $LUX$ to verify your answer. 

        $$
        \begin{equation}
        L = \left[ \begin{array}{ccc} 1 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & -1 & 1  \end{array}\right] \hspace{2cm} 
        U = \left[ \begin{array}{ccc} 1 & -1 & 0 \\ 0 & 1 & -1  \\ 0 & 0 & 1  \end{array}\right] \hspace{2cm} 
        B = \left[ \begin{array}{c} 2 \\ -3 \\ 4  \end{array}\right] 
        \end{equation}
        $$
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
        **Exercise 3:** Write a function called $\texttt{ForwardSubstitution}$ that will solve a lower triangular system $LY=B$.  It will be helpful to go back and look at the code for $\texttt{BackSubstitution}$.
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
        **Exercise 4:** Let $A$, $B$, and $C$ be the following matrices:

          $$
        \begin{equation}
        A = \left[ \begin{array}{ccc} 1 & 2 & 4 \\ 2 & 1 & 3 \\ 1 & 0 & 2  \end{array}\right] \hspace{2cm} 
        B = \left[ \begin{array}{ccc} 1 & 2 & 4 \\ 2 & 1 & 3  \\ 2 & 2 & 6  \end{array}\right] \hspace{2cm} 
        C = \left[ \begin{array}{ccc} 1 & 2 & 4 \\ 0 & -1 & -3  \\ 2 & 2 & 6  \end{array}\right] 
        \end{equation}
        $$

        $(a)$ Find an elementary matrix $E$ such that $EA = B$.  Verify with a computation.

        $(b)$ Find an elementary matrix $F$ such that $ FB = C$.  Verify with a computation.
            
        """
    )
    return


@app.cell
def _(lag, np):
    A_14 = np.array([[1, 2, 4], [2, 1, 3], [1, 0, 2]])
    I = np.eye(3)
    print('A: \n', A_14, '\n')
    B_6 = lag.RowAdd(A_14, 0, 2, 1)
    E_2 = lag.RowAdd(I, 0, 2, 1)
    print('B: \n', B_6, '\n')
    print('E: \n', E_2, '\n')
    print('EA: \n', E_2 @ A_14, '\n')
    B_6 = np.array([[1, 2, 4], [2, 1, 3], [2, 2, 6]])
    C_4 = lag.RowAdd(B_6, 2, 1, -1)
    F_1 = lag.RowAdd(I, 2, 1, -1)
    print('C: \n', C_4, '\n')
    print('F: \n', F_1, '\n')
    print('FB: \n', F_1 @ B_6, '\n')
    return A_14, B_6, C_4, E_2, F_1, I


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Consider the following $3\times 3$  matrix :

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr}  2 & 1 & 1\\  6 & 4 & 5  \\ 4 & 1 & 3 \end{array} \right] 
        \end{equation}
        $$

        $(a)$ Find **elementary matrices** $E_1$, $E_2$, and $E_3$ such that $ E_3E_2E_1A = U $ where $U$ is an upper triangular matrix.

        $(b)$ Find $L$ using the inverses of $E_1$, $E_2$, $E_3$, and verify that $  A = LU $.
        """
    )
    return


@app.cell
def _(lag, np, sla):
    A_15 = np.array([[2, 1, 1], [6, 4, 5], [4, 1, 3]])
    I_1 = np.eye(3)
    print('A: \n', A_15, '\n')
    B1_3 = lag.RowAdd(A_15, 0, 1, -3)
    E1_1 = lag.RowAdd(I_1, 0, 1, -3)
    B2_3 = lag.RowAdd(B1_3, 0, 2, -2)
    E2_1 = lag.RowAdd(I_1, 0, 2, -2)
    U = lag.RowAdd(B2_3, 1, 2, 1)
    E3_1 = lag.RowAdd(I_1, 1, 2, 1)
    print('E1: \n', E1_1, '\n')
    print('E2: \n', E2_1, '\n')
    print('E3: \n', E3_1, '\n')
    print('U: \n', U, '\n')
    L1 = sla.inv(E1_1)
    L2 = sla.inv(E2_1)
    L3 = sla.inv(E3_1)
    L_1 = L1 @ L2 @ L3
    print('L: \n', L_1, '\n')
    print('LU: \n', L_1 @ U, '\n')
    return A_15, B1_3, B2_3, E1_1, E2_1, E3_1, I_1, L1, L2, L3, L_1, U


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:**  Compute the $LDU$ factorization of the following matrix and verify that $A = LDU$.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr}  1 & 1 & 1\\  3 & 5 & 6  \\ -2 & 2 & 7 \end{array} \right] 
        \end{equation}
        $$

        """
    )
    return


@app.cell
def _(lag, np, sla):
    A_16 = np.array([[1, 1, 1], [3, 5, 6], [-2, 2, 7]])
    I_2 = np.eye(3)
    print('A: \n', A_16, '\n')
    B1_4 = lag.RowAdd(A_16, 0, 1, -3)
    E1_2 = lag.RowAdd(I_2, 0, 1, -3)
    B2_4 = lag.RowAdd(B1_4, 0, 2, 2)
    E2_2 = lag.RowAdd(I_2, 0, 2, 2)
    B3_1 = lag.RowAdd(B2_4, 1, 2, -2)
    E3_2 = lag.RowAdd(I_2, 1, 2, -2)
    print('B3: \n', B3_1, '\n')
    L1_1 = sla.inv(E1_2)
    L2_1 = sla.inv(E2_2)
    L3_1 = sla.inv(E3_2)
    L_2 = L1_1 @ L2_1 @ L3_1
    U1 = lag.RowScale(B3_1, 1, 1 / 2)
    U_1 = lag.RowScale(U1, 2, 1 / 3)
    D_1 = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    print('L: \n', L_2, '\n')
    print('U: \n', U_1, '\n')
    print('D: \n', D_1, '\n')
    print('LDU: \n', L_2 @ D_1 @ U_1, '\n')
    return (
        A_16,
        B1_4,
        B2_4,
        B3_1,
        D_1,
        E1_2,
        E2_2,
        E3_2,
        I_2,
        L1_1,
        L2_1,
        L3_1,
        L_2,
        U1,
        U_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Find $P, L,$ and $U$ such that $PA = LU$.  Following the discussion in this section, $P$ should be a permutation matrix, $L$ should be a lower triangular matrix with ones long the main diagonal, and $U$ should be an upper triangular matrix.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr}  1 & 3 & 2\\  -2 & -6 & 1  \\ 2 & 5 & 7 \end{array} \right] 
        \end{equation}
        $$
            
        """
    )
    return


@app.cell
def _(lag, np, sla):
    A_17 = np.array([[1, 3, 2], [-2, -6, 1], [2, 5, 7]])
    I_3 = np.eye(3)
    B1_5 = lag.RowAdd(A_17, 0, 1, 2)
    E1_3 = lag.RowAdd(I_3, 0, 1, 2)
    B2_5 = lag.RowAdd(B1_5, 0, 2, -2)
    E2_3 = lag.RowAdd(I_3, 0, 2, -2)
    print('B1: \n', B1_5, '\n')
    print('B2: \n', B2_5, '\n')
    P_1 = lag.RowSwap(I_3, 1, 2)
    print('P: \n', P_1, '\n')
    print('PA: \n', P_1 @ A_17, '\n')
    B_7 = lag.RowAdd(P_1 @ A_17, 0, 1, -2)
    F1 = lag.RowAdd(I_3, 0, 1, -2)
    print('B: \n', B_7, '\n')
    U_2 = lag.RowAdd(B_7, 0, 2, 2)
    F2 = lag.RowAdd(I_3, 0, 2, 2)
    print('U: \n', U_2, '\n')
    L1_2 = sla.inv(F1)
    L2_2 = sla.inv(F2)
    L_3 = L1_2 @ L2_2
    print('L: \n', L_3, '\n')
    print('LU: \n', L_3 @ U_2, '\n')
    return (
        A_17,
        B1_5,
        B2_5,
        B_7,
        E1_3,
        E2_3,
        F1,
        F2,
        I_3,
        L1_2,
        L2_2,
        L_3,
        P_1,
        U_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Use SciPy to compute the $PLU$ factorization of a $3\times 3$ matrix.  Replicate the results using the row operations functions in $\texttt{laguide}$.
        """
    )
    return


@app.cell
def _():
    ## Code Solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Applications
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Interpolation

        **Exercise 1:** In 2017, researchers from the Universities of British Columbia, Alberta, and Toronto published their findings regarding the population of snowshoe hares around Kluane Lake, Yukon. They measured the density of hares per hectare, taking a reading once every two years. Here are some of their measurements:

        | Measurement #    | Density per ha. |
        | ---------------- | --------------- |
        | 1                | 0.26            |
        | 2                | 0.20            |
        | 3                | 1.17            |
        | 4                | 2.65            |
        | 5                | 0.14            |
        | 6                | 0.42            |
        | 7                | 1.65            |
        | 8                | 2.73            |
        | 9                | 0.09            |
        | 10               | 0.21            |

        $(a)$ Find the unique ninth degree polynomial whose graph passes through each of these points.  Plot the data points together with the graph of the polynomial to observe the fit.
           
        $(b)$ Using the polynomial that we found, what should we expect the density of hares to be if we measured their population in the year between the third and fourth measurement? What about between the fourth and the fifth?
         
        $(c)$ Why might this method of interpolation not be an appropriate model of our data over time?
        """
    )
    return


@app.cell
def _(lag, np, plt):
    x_7 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([0.26, 0.2, 1.17, 2.65, 0.14, 0.42, 1.65, 2.73, 0.09, 0.21])
    A_18 = np.zeros((10, 10), dtype=int)
    B_8 = np.zeros((10, 1))
    for i in range(10):
        B_8[i, 0] = y[i]
        for j in range(10):
            A_18[i, j] = x_7[i] ** j
    coeffs = lag.SolveSystem(A_18, B_8)
    x_fit = np.linspace(x_7[0], x_7[9], 100)
    y_fit = coeffs[0] + coeffs[1] * x_fit + coeffs[2] * x_fit ** 2 + coeffs[3] * x_fit ** 3 + coeffs[4] * x_fit ** 4 + coeffs[5] * x_fit ** 5 + coeffs[6] * x_fit ** 6 + coeffs[7] * x_fit ** 7 + coeffs[8] * x_fit ** 8 + coeffs[9] * x_fit ** 9
    fig_2, ax_2 = plt.subplots()
    ax_2.scatter(x_7, y, color='red')
    ax_2.plot(x_fit, y_fit, 'b')
    ax_2.set_xlim(0, 11)
    ax_2.set_ylim(-7, 7)
    ax_2.set_xlabel('x')
    ax_2.set_ylabel('y')
    ax_2.grid(True)
    return A_18, B_8, ax_2, coeffs, fig_2, i, j, x_7, x_fit, y, y_fit


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Judging by the graph of the polynomial, it looks like the curve takes on a value of about 2.5 when $x = 3.5$ and a value of about 1 when $x = 4.5$.

        Although this method of finding a polynomial that passes through all our data points provides a nice smooth curve, it doesn't match our intuition of what the curve should look like. The curve drops below zero between the ninth and tenth measurement, indicating a negative density. But this is impossible, since we can't have a negative number of hares! It also rises much higher between the first and second measurement than the data points alone seem to suggest.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** The further you travel out into the solar system and away from the Sun, the slower an object must be travelling to remain in its orbit. Here are the average radii of the orbits of the planets in our solar system, and their average orbital velocity around the Sun.

        |Planet                           | Distance from Sun (million km)  | Orbital Velocity (km/s)         |
        | ------------------------------- | ------------------------------- | ------------------------------- |
        |Mercury                          | 57.9                            | 47.4                            |
        |Venus                            | 108.2                           | 35.0                            | 
        |Earth                            | 149.6                           | 29.8                            |
        |Mars                             | 228.0                           | 24.1                            |
        |Jupiter                          | 778.5                           | 13.1                            |
        |Saturn                           | 1432.0                          | 9.7                             |
        |Uranus                           | 2867.0                          | 6.8                             |
        |Neptune                          | 4515.0                          | 5.4                             |

        $(a)$ Find the unique first degree polynomial whose graph passes through points defined by Mercury and Jupiter.  Plot the data points together with the graph of the polynomial to observe the fit. Amend your polynomial and graph by adding Saturn, and then Earth. What do you notice as you add more points? What if you had started with different planets?
           
        $(b)$ Expand your work in part $(a)$ to a seventh degree poynomial that passes through all eight planets. The first object in the Kuiper Belt, Ceres, was discovered by Giuseppe Piazzi in 1801. Ceres has an average distance from the sun of 413.5 million km. Based on the points on the graph, estimate the orbital velocity of Ceres. What does the polynomial suggest the value would be?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We start by finding the polynomial and graph that passes through the points defined by Mercury and Jupiter.
        """
    )
    return


@app.cell
def _(lag, np, plt):
    x_8 = np.array([57.9, 778.5])
    y_1 = np.array([47.4, 13.1])
    A_19 = np.zeros((2, 2))
    B_9 = np.zeros((2, 1))
    for i_1 in range(2):
        B_9[i_1, 0] = y_1[i_1]
        for j_1 in range(2):
            A_19[i_1, j_1] = x_8[i_1] ** j_1
    coeffs_1 = lag.SolveSystem(A_19, B_9)
    x_fit_1 = np.linspace(x_8[0], x_8[1])
    y_fit_1 = coeffs_1[0] + coeffs_1[1] * x_fit_1
    fig_3, ax_3 = plt.subplots()
    ax_3.scatter(x_8, y_1, color='red')
    ax_3.plot(x_fit_1, y_fit_1, 'b')
    ax_3.set_xlim(0, 5000)
    ax_3.set_ylim(0, 50)
    ax_3.set_xlabel('Distance from sun (million km)')
    ax_3.set_ylabel('Orbital velocity (km/s)')
    ax_3.grid(True)
    return (
        A_19,
        B_9,
        ax_3,
        coeffs_1,
        fig_3,
        i_1,
        j_1,
        x_8,
        x_fit_1,
        y_1,
        y_fit_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next we add the point defined by Saturn.
        """
    )
    return


@app.cell
def _(lag, np, plt):
    x_9 = np.array([57.9, 778.5, 1432.0])
    y_2 = np.array([47.4, 13.1, 9.7])
    A_20 = np.zeros((3, 3))
    B_10 = np.zeros((3, 1))
    for i_2 in range(3):
        B_10[i_2, 0] = y_2[i_2]
        for j_2 in range(3):
            A_20[i_2, j_2] = x_9[i_2] ** j_2
    coeffs_2 = lag.SolveSystem(A_20, B_10)
    x_fit_2 = np.linspace(x_9[0], x_9[2])
    y_fit_2 = coeffs_2[0] + coeffs_2[1] * x_fit_2 + coeffs_2[2] * x_fit_2 ** 2
    fig_4, ax_4 = plt.subplots()
    ax_4.scatter(x_9, y_2, color='red')
    ax_4.plot(x_fit_2, y_fit_2, 'b')
    ax_4.set_xlim(0, 5000)
    ax_4.set_ylim(0, 50)
    ax_4.set_xlabel('Distance from sun (million km)')
    ax_4.set_ylabel('Orbital velocity (km/s)')
    ax_4.grid(True)
    return (
        A_20,
        B_10,
        ax_4,
        coeffs_2,
        fig_4,
        i_2,
        j_2,
        x_9,
        x_fit_2,
        y_2,
        y_fit_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next we add the point defined by Earth.
        """
    )
    return


@app.cell
def _(lag, np, plt):
    x_10 = np.array([57.9, 149.6, 778.5, 1432.0])
    y_3 = np.array([47.4, 29.8, 13.1, 9.7])
    A_21 = np.zeros((4, 4))
    B_11 = np.zeros((4, 1))
    for i_3 in range(4):
        B_11[i_3, 0] = y_3[i_3]
        for j_3 in range(4):
            A_21[i_3, j_3] = x_10[i_3] ** j_3
    coeffs_3 = lag.SolveSystem(A_21, B_11)
    x_fit_3 = np.linspace(x_10[0], x_10[3])
    y_fit_3 = coeffs_3[0] + coeffs_3[1] * x_fit_3 + coeffs_3[2] * x_fit_3 ** 2 + coeffs_3[3] * x_fit_3 ** 3
    fig_5, ax_5 = plt.subplots()
    ax_5.scatter(x_10, y_3, color='red')
    ax_5.plot(x_fit_3, y_fit_3, 'b')
    ax_5.set_xlim(0, 5000)
    ax_5.set_ylim(0, 50)
    ax_5.set_xlabel('Distance from sun (million km)')
    ax_5.set_ylabel('Orbital velocity (km/s)')
    ax_5.grid(True)
    return (
        A_21,
        B_11,
        ax_5,
        coeffs_3,
        fig_5,
        i_3,
        j_3,
        x_10,
        x_fit_3,
        y_3,
        y_fit_3,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The polynomial seems to have a smoother, more "natural" curve when there are fewer points, but quickly overshoots as more points are added. Finally we extend the polynomial to fill all the points in the table.
        """
    )
    return


@app.cell
def _(lag, np, plt):
    x_11 = np.array([57.9, 108.2, 149.6, 228.0, 778.5, 1432.0, 2867.0, 4515.0])
    y_4 = np.array([47.4, 35.0, 29.8, 24.1, 13.1, 9.7, 6.8, 5.4])
    A_22 = np.zeros((8, 8))
    B_12 = np.zeros((8, 1))
    for i_4 in range(8):
        B_12[i_4, 0] = y_4[i_4]
        for j_4 in range(8):
            A_22[i_4, j_4] = x_11[i_4] ** j_4
    coeffs_4 = lag.SolveSystem(A_22, B_12)
    x_fit_4 = np.linspace(x_11[0], x_11[7])
    y_fit_4 = coeffs_4[0] + coeffs_4[1] * x_fit_4 + coeffs_4[2] * x_fit_4 ** 2 + coeffs_4[3] * x_fit_4 ** 3 + coeffs_4[4] * x_fit_4 ** 4 + coeffs_4[5] * x_fit_4 ** 5 + coeffs_4[6] * x_fit_4 ** 6 + coeffs_4[7] * x_fit_4 ** 7
    fig_6, ax_6 = plt.subplots()
    ax_6.scatter(x_11, y_4, color='red')
    ax_6.plot(x_fit_4, y_fit_4, 'b')
    ax_6.set_xlim(0, 5000)
    ax_6.set_ylim(0, 50)
    ax_6.set_xlabel('Distance from sun (million km)')
    ax_6.set_ylabel('Orbital velocity (km/s)')
    ax_6.grid(True)
    return (
        A_22,
        B_12,
        ax_6,
        coeffs_4,
        fig_6,
        i_4,
        j_4,
        x_11,
        x_fit_4,
        y_4,
        y_fit_4,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The polynomial looks like it takes on a value near 0 when $x = 413.5$, but the points suggest something closer to 18.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
