"""
A "pipeline" is a name for the processes you go through when solving problems, every type of work can have a pipeline,
not just ML. This file will have some notes on how to go about solving data science problems, as well as some useful
functions that I am likely to need regularly.

A typical workflow can go like this:

1. Business problem
    Your boss could say "Hey, we've got a question for you, how do we <keep our customers happy>?

2. Data Acquisition
    For whatever the problem is, think about what kind of data would be useful, what's critical and what's a bonus
    Where do you get it from? A database? Web scraping? Old paper files that need to be digitised? Whatever you can.

3. Data Preperation
    It's wishful thinking to presume that the data will be already nicely organised, consistent and complete.
    A lot of the grunt work as a data scientist is preparing/preprocessing the data. Get comfortable with it!
    This would involve doing stuff like one-hot encoding categories, judging if missing data is too problematic,
    imputing (filling missing values with appropriate dummies), amalgamating different data sources and making them
    all consistent. All the "admin" type work that when done well, makes the rest of the task go a lot more smoothly.

4. Data Analysis
    Now we get to be big-brain. Probably start with some exploratory data analysis (EDA) to play about with the data,
    make some trivial graphs, throw out a correlation matrix, just explore what you have, ponder the relationships,
    remember what it is you are actually trying to answer. This is more of a thinking procedure.
    It is important that you stay both logical and sensible: It's easy to just slap data together with no true insight.

5. Data modelling
    This is the fun part. This is where you already:
    1. Know what problem you're solving
    2. Have gathered data to help your solve it
    3. Have prepared the data so that you can use it effectively
    4. Have focused in on how you will use that data
    Now you build models, use math, use algorithms, make graphs, produce insights, and answer the problem.

6. Visualisation and Communication
    Now that you have used your skills as a data scientist to deeply understand and solve the problem you had,
    it's time to share that knowledge with the world. Make things pretty, clear, and explain what you have found.
    You are likely to be presenting these findings to non-technical people, those who need your skills and know-how.
    There's no point in solving problems and then having the solution be incomprehensible, you should be able to
    explain it to anyone who is interested, no matter their understanding of your work.

7. Deployment and maintenance
    After a write-up or presentation, it will be time to wrap up the project and go on to the next one.
    But it is important to leave your findings in excellent condition for the next people who may be interested.
    Your solution may prove useful in later projects and you will thank yourself for leaving it in such a good state.
    A streamlit website, github readme, notes and documentation, interactive graphs, whatever you want really!

And that's a basic outline of how a project could go - it's good to break down projects into steps, and sub-steps.
----------
Now for a machine learning pipeline, a helpful function that will allow you to work more effectively.


"""
