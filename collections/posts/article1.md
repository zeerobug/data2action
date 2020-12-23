---
author: Thomas Vié
date: Created
image: /_includes/assets/img/network.jpg
title: Similarities in a multidimensional dataset
subTitle: A fast and easy way to find the perfect match
layout: layouts/article
link: blog/article1
category: Technology
excerpt: 'Most of data science is oriented towards the Train, Test, Predict paradigm. But there are some cases where other implementations are needed like unsupervised classification or discovering patterns in existing data.'
---

<img alt="Image for post" class="et fg fc ix v" src="{{image}}" width="100%"/>

Photo by [Alina Grubnyak](https://unsplash.com/@alinnnaaaa?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://medium.com/s/photos/network?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# A real life example

Most of data science is oriented towards the **Train, Test, Predict paradigm**. Who doesn’t want to guess the future! But there are some cases where other implementations are needed like unsupervised classification or discovering patterns in existing data. In other words, how to take advantage of the **data which is already owned**.

I think this aspect is a little bit **disregarded** and the literature about it is scarcer, compared to other branches of data science. Hence the reason for this little contribution.

Here’s the story: A client of ours needed a way to find similar items (neighbours) for a given entity, according to a fixed number of parameters. Practically, the dataset is composed of votes from Human Resources Professionals who could attribute **up to 5 skills** to an arbitrary amount of World universities. It means that Edouard from HR could vote for MIT as a good institution for Digitalisation, Oxford for Internationality and La Sorbonne for Soft Skills.

I prepared the data, output a Spiderweb chart where the client could choose any Institution and compare it with the others, here is an example for three random universities:

<center>
<img alt="Image for post" class="pb-2" src="/_includes/assets/img/original.png" width="70%"/><br/>
Voted skills for three universities
</center>
At that point, it seemed interesting to search for universities that would have been voted the same way, maybe to compare their actions and study what they were doing good and what wrong.

The data came in a spss file, with one row by vote, and according to our brief, the output had to be fast, because it was meant to be used as a Backend service, with near **real time responses**.

After some research, I thought that the best processing format for that would be a [**KD Tree**](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html) for its multi-dimensional nature and its relatively easy and fast processing possibilities. I won’t explain in detail what KD Trees are but you can refer to the [wikipedia article](https://en.wikipedia.org/wiki/K-d_tree).

It is fully integrated into the [**sklearn**](https://scikit-learn.org/stable/) module, and very easy to use as we’ll see below.

But first, let's do some prep!

# Data Preparation

Our dataset, as property of the client, has been anonymised. The names of the universities have been taken away, but the values are real.

We’ll start by importing the libraries:

```python
import pandas as pd
from sklearn.neighbors import KDTree
```

- Pandas will be used to import and process our data. It is very fast and useful for database-like processing
- sklearn stands for [scikit-learn](https://scikit-learn.org/stable/), one the most famous library for data analysis. It is used for classification, clustering, regression and more. We’ll just import KDTree from the **Nearest Neighbors** sub-library

We already converted the spss file to a csv file, so we just have to import it using pandas read_csv method and display its first rows:

```
df = pd.read_csv("[https://bitbucket.org/zeerobug/material/raw/12301e73580ce6b623a7a6fd601c08549eed1c45/datasets/votes_skills_anon.csv](https://bitbucket.org/zeerobug/material/raw/12301e73580ce6b623a7a6fd601c08549eed1c45/datasets/votes_skills_anon.csv)", index_col=0)
df.head()
```

Dataset structure

Each row corresponds to a vote where:

- **SUID** is the Unique ID of the voter
- **univid**: the unique ID of the institution
- **response**: the voted skill

So for example that means the user #538 voted “Internationality” as an important skill for University #5c9e7a6834cda4f3677e662b.

Our next step consists in grouping by institution and skill (response). We do it with the excellent **groupby** method that generates a SeriesGroupBy object that we can use to count the number of similar pairs of (univid, response) in the dataset.

```
skills = df.groupby(["univid", "response"])["response"].count().reset_index(name='value')
```

We use reset_index, to get a DataFrame back from the series which is output by the count() function, and to create the “value” column that contains that count. Here is our table now:

<center>
<img alt="Image for post" class="et fg fc ix v" src="/_includes/assets/img/table.png" width="70%"/>

Same dataset grouped by univid and response

</center>
Even if a **lot more readable**, this format is useless for our goal. It is difficult to distinguish between institutions as the number of rows is arbitrary (some skills may have not been voted), and lots of tools work best with row values instead of columns values.

Luckily, Pandas offers a very powerful tool which swaps rows and columns: [**pivot_table**](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html). Its arguments are self explanatory so I won’t enter into details.

```
univSkills = skills.pivot_table(values="value", columns="response", index=["univid"])
univSkills.head()
```

<center>

<img alt="Image for post" class="et fg fc ix v" src="/_includes/assets/img/table2.png" width="100%"/><br/>
Values after pivot_table

</center>

Our data is almost ready for processing, but we still have an issue: To be **comparable**, each row must be in the same range and if we calculate the sum of values in a row, the total is far from being similar from one row to another:

```
univSkills.sum(axis=1).head()univid
5c9e345f34cda4f3677e1047    69.0
5c9e349434cda4f3677e109f    51.0
5c9e34a834cda4f3677e10bd    40.0
5c9e34ae34cda4f3677e10c7    66.0
5c9e34d534cda4f3677e1107    70.0
```

This is because universities like Harvard have had a lot more votes that some remote and unknown university. It could be interesting to use that parameter in some other calculation but for the present problem, we need to get rid of that **disparity**. We comply by using **percentages** instead of **absolute values.**

So we have to sum each line and divide each value by that sum. This is done in a **one-liner**, and then we get rid of some Nan values to finish polishing the dataset.

```
univSkills = univSkills.div(univSkills.sum(axis=1), axis=0)
univSkills.fillna(0, inplace=True )
```

Our dataset is now clean, ready, the values are in the same range, we can start playing with some more interesting processing.

# Processing the data to find neighbors

So our algorithm has to detect amongst all the universities in our dataset which ones have the closest values to our 5 variables at the same time. We can immediately think of a **brute force algorithm** with nested loops that would compare value by value until it finds the 5 closest values for each variable but that would be **far from optimal** and not fast enough for our appplication!

The KD Tree algorithm is way more effective, it consists of a **geometrical approach** of the data, which, by subsequent divisions of a n-dimensional space, generates a tree that **re-organises the data** in a way that allows **complex queries** to run very fast. So let’s generate a tree with that method:

```
tree = KDTree(univSkills)
```

Our tree is ready to be queried. The first step consists in choosing a university to start with, for example the row of index 9 (`univSkills[9:10]`), we want a result set of the 5 closest universities (`k=5`) and the "query" function applied to our tree will return a tuple of 2 numpy arrays (`dist, index`), which will be respectively the **distance** and the **index** of the result, sorted from the closest to the furthest.

```
dist, ind = tree.query(univSkills[9:10], k=5)
```

And then we can display the values of our **neighbors**:

```
univSkills.iloc[ind.tolist()[0]]
```

We notice right away that **the values are very close**, we can confirm it with a new Radar chart:

<center>
<img alt="Image for post" class="et fg fc ix v" src="/_includes/assets/img/match1.png" width="70%"/><br/>
5 University skills compared

</center>
<p></p>
<center>
<img alt="Image for post" class="et fg fc ix v" src="/_includes/assets/img/match2.png" width="70%"/><br/>
Another example

</center>

You can try with different starting row, in most of the cases, the radar charts will remain very **similar in shape**.

You can also play with the other **KDTree method variables** that you will find in the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html#sklearn.neighbors.KDTree.query). Let me know if your experiments lead to better results.

# Further experiments

I think there are a lot of other applications of this algorithm. It can be used in **recommender systems**, **dating sites**, and generally any processing that relies on proximity between multidimensional matrices

**Relationship**, for example: by determining a maximum distance and apply the query function to each row of the whole dataset, we could spot unseen relations and generate a **GraphQL**-like database.

Due to its speed, simplicity and effectivity, the **KDTree** can also be employed in some simple cases as a replacement for far more complicated libraries like **TensorFlow** or **Pytorch**. We’ll look into that in my n**ext article**.

Et voilà! I hope that this article will be of use to someone. You can find the complete notebook [here](https://jovian.ml/zeerobug/article-calculate-skills-similarity). Don’t hesitate to leave a comment and send me an email, or message for any inquiry.
