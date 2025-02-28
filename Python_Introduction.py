import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to Python
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This section includes many cells which contain Python code.  The output from each cell is visible directly below the cell.  We provide an explanation for each piece of code as well as some suggestions for the reader to experiment with.  In order to engage with the notebook and see new results, edit the code cell, then press 'Shift + Enter'.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Calculations, variable assignments, and printing

        The first thing we might try is the use of code cells as a calculator.  Indeed this works in a straightforward way.  Run the cell below to see how it works.  Experiment with the input to see how to enter various operations.  (Note that you can get exponents by using the \*\* operator.)  
        """
    )
    return


@app.cell
def _():
    42*1.2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The next thing to learn is how to assign names to numbers and other objects.  For this task we use a command like $\texttt{b=32/5}$.  This code computes the value of $\texttt{32/5}$, stores this value as an object, and then assigns the name $\texttt{b}$ to that object.
        """
    )
    return


@app.cell
def _():
    b = 32/5
    return (b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since the result was stored, the actual value of $\texttt{b}$ is not displayed as ouput from the code cell.  Indeed, nothing is displayed when the cell is executed.  Typically, when we want to display the results of a computation, we have to use a command called $\texttt{print}$.
        """
    )
    return


@app.cell
def _():
    b_1 = 32 / 5
    print(b_1)
    return (b_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we print the results of many calculations, it becomes difficult to determine which number is which.  We will often want to print text with numerical results to provide descriptions.  The text to be printed must be placed in quotes.  It is also possible to provide multiple items to the $\texttt{print}$ command by separating them with commas.
        """
    )
    return


@app.cell
def _(b_1):
    print('The calculation is complete.')
    print('The result of the calculation 32/5 is', b_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The text within the quotes is a Python object known as a **string**.  Just as with the value of $\texttt{b}$, we could assign the string object a name if we plan to reuse it.  Python offers many powerful ways to manipulate and process strings, but our primary use of strings is to display informative messages about our numerical results.  
        """
    )
    return


@app.cell
def _(b_1):
    result_message = 'The result of the calculation is'
    print(result_message, b_1)
    print(result_message, b_1 * 37)
    return (result_message,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It is sometimes easier to read output if there are extra space to break up the lines.  We frequently print out the special string $\texttt{'\n'}$, which adds an extra line break. 
        """
    )
    return


@app.cell
def _(b_1, result_message):
    print(result_message, b_1, '\n')
    print(result_message, b_1 * 37)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The $\texttt{print}$ command offers several other ways to manipulate the basic format of the output it produces.  We will comment on these options as they are used.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Functions

        Quite often we write a bit of code that we would like to reuse.  Maybe it was tricky to figure out, or just tedious to type, and we want to save time and effort by making use of the code we've already written.  We could just copy and paste code from here to there, but that quickly becomes a chore as the amount of code we need continues to grow.  

        One simple way to reuse code is to define a **function**.  We can think of a function as a new command that will carry out whatever instructions we would like to include.  Sometimes we will find it useful to provide the function with information in order for it to do its job.  Other times we might ask that the function return some information to us when it has finished its job.  Let's look at a couple of examples.

        In the first example, we won't exchange information with the function, we will just ask it to display a message.  To define a function we use the keyword $\texttt{def}$.  We then list any instructions that we want the function to carry out.  The $\texttt{def}$ command must end with a colon (:), and all instructions that are part of the function **must be indented** with a 'Tab' key. 
        """
    )
    return


@app.cell
def _():
    def InchConversionFactor():
        print("There are 2.54 centimeters in 1 inch.")
    return (InchConversionFactor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that when we execute the cell, the message is not printed.  This statement only defines the function.  In order to execute the commands in the function, we have to call it.
        """
    )
    return


@app.cell
def _(InchConversionFactor):
    InchConversionFactor()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When we call the function, we need to include the parentheses () as part of the call.  In this example the parentheses are empty because we are not passing any information to the function.  Let's write another function that allows us to provide it with a measurement in inches, and have it print out the measurement in centimeters.
        """
    )
    return


@app.cell
def _():
    def InchtoCentimeterConversion(inches):
        cm = inches*2.54
        print(inches,"inches equals",cm,"centimeters.")
    return (InchtoCentimeterConversion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Again, nothing actually happens until we call the function.  This time when we call though, we will need to provide a number that the function will interpret as the variable $\texttt{inches}$.  The objects that we pass into functions are known as **arguments**.
        """
    )
    return


@app.cell
def _(InchtoCentimeterConversion):
    InchtoCentimeterConversion(2.3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the final example we will provide the function with the measurement in centimeters, and it will *return to us* the measurement in centimeters without printing anything.
        """
    )
    return


@app.cell
def _():
    def ReturnInchtoCentimeterConversion(inches):
        cm = inches*2.54
        return cm
    return (ReturnInchtoCentimeterConversion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And again we must call the function to carry out the code.  This time however, the function creates an object that represents the value being returned.  In order to make use of this object we must assign it a name as before.
        """
    )
    return


@app.cell
def _(ReturnInchtoCentimeterConversion):
    result = ReturnInchtoCentimeterConversion(2.3)
    print("Our result is",result,"centimeters.")
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Conditional statements

        It is quite common that we will want some commands to be carried out only in certain conditions.  If we are carrying out division for example, we might want to be certain we are not dividing by zero.  The $\texttt{if}$ keyword lets us check a condition before proceeding with any associated commands.  The structure of an *if block* is similar to that of a function definition.  The $\texttt{if}$ keyword is followed with a condition and a colon, then all associated commands **are indented** to indicate that they are only to be executed when the condition is true.
        """
    )
    return


@app.cell
def _():
    _a = 2
    b_2 = 5
    _c = 0
    if b_2 != 0:
        result1 = _a / b_2
        print(result1)
    if _c != 0:
        result2 = _a / _c
        print(result2)
    return b_2, result1, result2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this case only the value of $\texttt{result1}$ was computed.  The condition in the second if block ($c$ not equal to 0) is not true, so the instructions in this block are not executed.  Note that the conditions have their own precise syntax as well.  Here are a few common conditions.

        - $\texttt{a > b}$    (Is $a$ greater than $b$?)
        - $\texttt{a == b}$   (Does $a$ equal $b$?)
        - $\text{a != b}$   (Does $a$ not equal $b$?)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Sometimes we will want some commands executed if a condition is true, but *other commands* executed if the condition is not true.  In this scenario we use the $\texttt{else}$ keyword to define an *else block*, which forms the alternative to the *if block*.  Let's suppose $a$ has been assigned a value, and we want to compute $|a|$, the absolute value of $a$.
        """
    )
    return


@app.cell
def _():
    _a = -8
    if _a > 0:
        abs_value = _a
    else:
        abs_value = -_a
    print('The absolute value of a is', abs_value)
    return (abs_value,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Try changing the value of $a$.  What happens if $a=0$ ? 

        At other times we will want to *repeat* commands while a condition is true.  This is done with the $\texttt{while}$ command, and a corresponding *while block* of code.  We demonstrate with code that totals all the integers from $a$ to $b$.  
        """
    )
    return


@app.cell
def _():
    _a = 3
    b_3 = 20
    count = _a
    _sum = 0
    while count <= b_3:
        _sum = _sum + count
        count = count + 1
    print('The total is', _sum)
    return b_3, count


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Blocks of code that get repeated like this are known as **loops**.  The commands in the *while block* are said to be "inside the loop". 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Iterations

        The most common type of loop that we will use is created with the $\texttt{for}$ keyword, and is thus known as a *for loop*.  The commands inside a *for loop* get repeated once for each object in a specified collection.  For our purposes, the collection of objects will almost always be a set of numbers generated using the $\texttt{range}$ command.  The combination of $\texttt{for}$ and $\texttt{range}$ is easiest to understand by looking at some examples.
        """
    )
    return


@app.cell
def _():
    for _number in range(5):
        print(_number)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this first example every number in $\texttt{range(5)}$ gets printed.  We can see that $\texttt{range(5)}$ contains the numbers from 0 to 4.  In general $\texttt{range(n)}$ will generate the numbers 0 to $n-1$.  It may appear strange that the collection is 0 to 4 instead of 1 to 5, but counts beginning with zero are common in programming languages.  We can specify any starting number we like by providing another argument to the $\texttt{range}$ function.
        """
    )
    return


@app.cell
def _():
    for _number in range(3, 10):
        print(_number)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can also provide the $\texttt{range}$ function a third argument to specify the *spacing* between numbers.
        """
    )
    return


@app.cell
def _():
    for _number in range(4, 12, 2):
        print(_number)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With $\texttt{for}$ and $\texttt{range}$ we can create another loop that totals the numbers from $a$ to $b$ as we did earlier using $\texttt{while}$.
        """
    )
    return


@app.cell
def _():
    _a = 3
    b_4 = 20
    _sum = 0
    for i in range(_a, b_4 + 1):
        _sum = _sum + i
    print('The total is', _sum)
    return b_4, i


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One difference between using a $\texttt{for}$ loop and a $\texttt{while}$ loop is that we do not need to explicitly increment a variable to track the number of iterations when using a $\texttt{for}$ loop.  In general, we tend to use a $\texttt{for}$ loop when we want to iterate once for each item in a collection, such as the numbers in $\texttt{range(5)}$. We use a $\texttt{while}$ loop when we want the number of iterations to remain flexible.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Comments

        Often it is useful to include some plain text inside the code cell to help provide a description or explanation.  Such text is called a **comment**.  Comments are not interpreted as Python code, so we can write anything we like to help us document the code.  There are two ways to accomplish this.

        - Any portion of a line that follows the $\texttt{#}$ symbol is ignored by the interpretor.

        """
    )
    return


@app.cell
def _():
    _a = 142
    _c = _a ** 2
    print('The square of', _a, 'is', _c)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Any portion of the code cell within triple quotes is ignored.
        """
    )
    return


@app.cell
def _():
    """ 
    We can write 
    several lines
    worth of comments to explain
    the purpose of the code
    """
    _a = 142
    _c = _a ** 2
    print('The square of', _a, 'is', _c)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The important point here is that the previous two cells do exactly the same thing.  The comments have no effect on the calculation or the output.  The following cell with no comments also does exactly the same thing.
        """
    )
    return


@app.cell
def _():
    _a = 142
    _c = _a ** 2
    print('The square of', _a, 'is', _c)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Importing libraries

        The core of the Python language does not contain all the functionality that we will need for linear algebra.  In order to access more tools we will **import** some Python modules.  For example, some basic functions that you might find on a scientific calculator are not part of the Python language, but are included in the math module.  A simple way to import this module is with the code $\texttt{import math}$.  The cell below shows some examples of how we might use the module.
        """
    )
    return


@app.cell
def _():
    import math
    s = math.sqrt(2)
    print("The square root of 2 is approximately",s)
    PI = math.pi
    print("Pi is approximately", PI)
    print("The cosine of pi/10 is approximately",math.cos(PI/10))
    return PI, math, s


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that when we use the square root function, we need to use the call $\texttt{math.sqrt}$ instead of just $\texttt{sqrt}$.  This is because the square root function is actually a part of the $\texttt{math}$ module, and is not in the basic set of Python commands.  The use of this dot notation is ubiquitous in Python.  Whenever $\texttt{object2}$ is contained within $\texttt{object1}$, we access $\texttt{object2}$ by using $\texttt{object1.object2}$.

        In the Jupyter Guide to Linear Algebra, we will create our own module that contains all of the functions we develop as we work through the material.  This module is named $\texttt{laguide}$.  We will import it in later sections when we want to make use of functions that we create in the earlier sections.  Note that unlike the $\texttt{math}$ module, the $\texttt{laguide}$ module is not in the Python standard library.  This means that in order to use $\texttt{laguide}$ outside of the Jupyter Guide to Linear Algebra, the source code must be copied from the repository [github.com/bvanderlei/jupyter-guide-to-linear-algebra](https://github.com/bvanderlei/jupyter-guide-to-linear-algebra) before it can be imported.      
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

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
        ### Resources

        - [A Crash Course in Python for Scientists](https://nbviewer.jupyter.org/gist/rpmuller/5920182)
        - Lubanovic, Bill. [Introducing Python](https://www.oreilly.com/library/view/introducing-python-2nd/9781492051374/), O'Reily Media, 2019
        - [Real Python](https://realpython.com/)
        - [The Python Tutorial](https://docs.python.org/3/tutorial/)

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
