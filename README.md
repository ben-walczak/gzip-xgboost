# Gzip and XGBoost

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

Where do semantic come into play? If we had the following two strings what would happen?
```
Sentence 1: The president spoke to the crowd in Chicago
Sentence 2: Obama addressed the audience in his hometown
```
Structurally, these sentences are not similar but semantically they are very similar, so how do we capture this if we are only measuring structural similarity? To be honest I'm not entirely sure. I would hope the downstream model captures some of the semantic relationships, but that's not quite true according to how we build our features. More work needs to be addressed here as well. But then again, do we even need semantic similarity for some of the problems we would like to solve? Just how much semantics actually benefits these models?