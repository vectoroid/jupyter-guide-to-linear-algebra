import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to Jupyter
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Introduction to Python
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Write a function that accepts 5 numbers as arguments and returns the average of those numbers.
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
        - Write a function that takes accepts a single integer and displays whether the integer is odd or even.
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
        - Write a function that accepts a single argument $N$, and returns the largest square number that is less than or equal to $N$.
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
        - Write a function that accepts two arguments, $\texttt{a}$ and $\texttt{b}$, and returns the remainder of $\texttt{a/b}$.  (*There is a built-in Python operator that does this, but try to come up with a way to do it for yourself.*)
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
        - Write a function that accepts a single integer $N$, as an argument, and returns the number of factors of $N$.   (*For example, 18 has factors 1, 2, 3, 6, 9, 18.  If the function receives 18 as an argument, it should return 6.*)
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
        - Write a function that accepts a single argument that represents a date, and returns the number of days that have passed between January 1, 2000, and the date provided.  (*For example, if the function receives the number 020100 (February 1, 2000), it should return the number 31.  If the function receives the number 01012001 (January 1, 2001), it should return 366 since the year 2000 was a [leap year](https://www.timeanddate.com/date/leapyear.html).*)  This exercise is more challenging that it may first appear.  Try splitting the problem in to simpler tasks.
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
        ### Introduction to Numpy and Matplotlib
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Using the $\texttt{random}$  module, create a NumPy array with 2 rows and 3 columns that has entries that are random positive integers less than 10.  Print your array. Change the entry in the first row, second column to 8 and double the value in the second row, third column. Print the resulting array to check that your code has made the correct modifications.
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
        - Using the $\texttt{random}$  module, create a NumPy array with 3 rows and 3 columns that has entries that are random positive integers less than 10.  Print your array. Now multiply each value in the matrix by 5 and then add 1 to all the enteries in the second row. After that, divide each diagonal entry by 2 and print the resulting array to check that your code has made the correct modifications.
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
        - Using the $\texttt{random}$  module, create a NumPy array with 2 rows and 3 columns that has entries that are random integers greater than 5 and less than 13.    Write code that sets all even entries of the matrix to zero.  Print the array and check the results.  Run the code several times to check that it works for *different* random arrays. 
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
        - Create a NumPy array with 2 rows and 4 columns and assign it the name A.  Fill it with some values of your choice.  Create a new array $B$ that has the columns of $A$ as it's rows.  Print $A$ and $B$.  Below is an example of one such pair of arrays.

        $$
        \begin{equation}
        A = \left[ \begin{array}{cccc} 1 & 0 & 5 & 1 \\ 3 & 2 & 0 & 8   \end{array}\right]\hspace{1cm}
        B = \left[ \begin{array}{cc} 1 & 3 \\ 0 & 2 \\ 5 & 0  \\ 1 & 8  \end{array}\right]\hspace{1cm}
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
        - Using matplotlib to plot the following points using the blue *cross* symbol: (2,1), (4,4), (8,16), (10,25).  Add a grid that consists of dotted lines as in Example 2. Set the limits of $x$-axis to 0 and 10.  Set the limits of the $y$-axis to 0 and 25.  The numbers and gridlines should appear at intervals of 2 units along the $x$-axis and at intervals of 5 units along the $y$-axis.  Label the $x$-axis as "*Time*" and the $y$-axis as "*Distance*".
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
        - Use matplotlib to plot the curve $y= 4+x-0.5x^2$ where $x$ lies in the interval $[-1,5]$.  Follow Example 2 and use $\texttt{linspace}$ to generate an array of coordinates of points on the curve. The curve should be dashed and shown in green color. The numbers and gridlines should appear at intervals of 1 unit along the $x$-axis and at intervals of 3 units along the $y$-axis. Draw the lines $x=0$, and $y=0$ with black colour and 1 unit width. Add a grid that consists of dashed lines.
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
