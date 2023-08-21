# Gzip and XGBoost

**Scroll down to Latest Update as this project no longer uses Gzip compression distance, but rather Levenschtein distance**

The paper "Less is More: Parameter-Free Text Classification with Gzip" introduced an interesting and novel way of using the difference or similarity of two strings, compression distance, in a KNN to be used in multi-class text classification. KNN is intuitive, explainable, and relatively small when compared to large BERT models used today. So, the paper provides a refreshing new lense on language problems. In this example, we break apart the shortest compression distances and count the number of instances of that distance occur for each class for k of the smallest compression distances to be used as features in a downstream model. Large compression distances are not as useful as small compression distances, therefore we will only consider the smallest k distances. I would like to blindly use the top k distances and their associated target class, but I found out during testing that many strings have the same distance to the new string being measured. Therefore, counting where this occurred seemed to be more valuable.

Admittedly, the biggest drawback of this approach is calculating the distance of a new string compared to all previously seen strings is extremely slow. This is done to both build the features for the training set and the testing set. Future work could include a modified version of KNN, so we would not have to iterate through the entire dataset again to calculate the distances. Or another approach could be parallelizing the calculation of these distances.

Interestingly, compression distances is only one measure of similarity between strings. There are other distances we could test out in the future:
- Levenshtein Distance
- Hamming Distance
- Jaccard Distance (similarity)
- Jaro-Winkler Distance
- Longest Common Subsequence
- Damerau-Levenshtein Distance
- Dice's Coefficient
- Monge-Erkan Measure

Where do semantics come into play? If we had the following two strings what would happen?
```
Sentence 1: The president spoke to the crowd in Chicago
Sentence 2: Obama addressed the audience in his hometown
```
Structurally, these sentences are not similar but semantically they are very similar, so how do we capture this if we are only measuring structural similarity? To be honest I'm not entirely sure. I would hope the downstream model captures some of the semantic relationships, but that's not quite true according to how we build our features. More work needs to be addressed here as well. But then again, do we even need semantic similarity for some of the problems we would like to solve? Just how much semantics actually benefits these models?

## Latest update

Well, long story short gzip is extremely slow on large datasets as it needs to calculate each compression distance of between each string. Even with parallelization, the algorithm still ran too slow for my short attention span. In my latest commit, I wanted to use compression distance of gzip from two strings, but I didn't want to make a custom implementation to work inside a `cuda.jit` decorated function. So, I opted to use another distance metric as a placeholder, `Levenschtein Distance`. I calculated the distance on 5000x5000 strings from the training dataset, where the 5000 strings were different from the other 5000 strings, and the second set of 5000 strings contained the labels. I then used this dataset of 5000x5000 distances to train an XGBoost model. Then I tested this on 100 strings from the test dataset. I know these are all small sample sizes compared to the 1.4 million rows of the training dataset and the 60,000 rows of the test set from the "Yahoo Question Answers" dataset, but this algorithm was uncomfortably slow and I simply just wanted a POC more than anything. From the 100 strings in the test dataset, I achieved 27% accuracy, far below the benchmark of other popular algorithms.

There are a lot of things I could have done differently to improve speed and accuracy:
- I could have used larger sample sizes
- I could have used better distance metrics, as from the limited POC Gzip compression distance *seems* better than Levenschtein Distance
- I could have minimized the examples used by transforming the data to another format for distance calculations, although I am unfamiliar with any transformation that could achieve this
- I could have minimized the examples used by looking at the feature importance from XGBoost, so we don't waste time on unnecessary distance calculations

If and when I do have more time to explore the interesting and novel way of using distance metrics between two strings such as compression distance and Levenschtein Distance, I think the next best idea to explore would be only using examples from the training dataset with the largest feature importance. But for now, I'm going to close the chapter on this small experiment and POC. I enjoyed working with this, but I now realize some of the glaring limitations with Gzip compression distance approach for multi-class classification.