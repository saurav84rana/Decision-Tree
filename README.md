# Decision Tree in C++

## Overview
This is an implementation of Decision Tree (best first) in C++, which extends to support Random Forest.
It trains on `training.dat` and tests on `testing.dat` to give an accuracy score.

## Design notes
As per original assignment's requirements:
- Single `.cpp` file only,
- The algorithm must train on `training.dat` and tests on `testing.dat`. This means that Random Forest cannot use OOB evaluation for results.


## Build and Run
> **Please note**: `training.dat` and `testing.dat` are not included.
```bash
g++ -std=c++17 DecisionTree.cpp -o forest
./forest
```

## Data files
The `.dat` files are structured such that each row has 10 features, followed by its class:
```
f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 class
```
Where:
- **fn** is nth feature. Each feature is a boolean (either 0 or 1), or -1 to mark as missing,
- **class** is either 0, 1, or 2.

Invalid values (any values not in {-1, 0, 1}), are treated as missing.

## Decision Tree
This decision tree repeatedly splits the data, picking the single best split across the entire tree at each step (best first growth). Each decision node (`RuleNode`) in the tree represents a decision (a condition on one of the feature), and leaves (`LeafNode`) represent classification outcomes.

**Handling missing values**: (`bestSplit()`)\
At each decision node, missing values (-1), or any values that is not 0 or 1, is routed left if `missingGoesLeft`, else right. This is decided during training; uses whichever gave the best split.

**In place tree growth (`Node**`)**: (`buildDecisionTree()`)\
When a leaf is chosen for splitting from min heap, it's replaced by directly writing into the pointer position that holds the leaf, via pointer to pointer reference.

### Node types:
**RuleNode**: Internal (Decision) node that holds a splitting rule based on a single feature, and pointer to 2 child nodes.

**LeafNode**: Terminal node containing indices of samples and a class prediction. 

Both use polymorphism, abstract Node class.

### Splitting Criteria (Gini Impurity): 
Gini index is used to evaluate how well a split separates the classes. Per decision node, it  tries all features and goes with the feature that gives the lowest value.

$$ Gini = 1 - \sum^C_{i=1} (p_i^2) $$

Where:  

$C$: Total number of classes. 3 for this dataset; either 0, 1, or 2, 

$p_i$: The proportion of data points in the node that belong to class $i$.


### Choose correct split (best first): 
- Uses a min-heap priority queue to store potential leaf splits (`SplitCandidate`).
- Each node’s impurity improvement is used as its priority. (Priority is the within node Gini reduction, not weighted by node size)
- Therefore, nodes with the largest reduction in impurity are split first, regardless of where the node is. 
- Stops growing after reaching `maxRuleNode`, or the if node is pure.

A split is accepted even if it doesn't immediately reduce impurity (unless leaf is already pure). This is because it may set up for later splits with better improvement.

## Random Forest
Random forest is an ensemble of decision trees. Each tree is trained on bootstrapped samples, and with random selected features allowed for splitting. 

Bootstraping means ~37% of each tree's training data are never used for that tree, which means it can be used for OOB (out of bag) evaluate instead of a separate test set. However, this implementation reports accuracy against `testing.dat` as per assignment requirement.

**Bootstrapping**: Each tree receives a bootstrapped sample of the data to train on. Bootstrapped sample is same size as dataset, but each item is picked from random, and allows duplicates.

**Feature Masking**: Each tree also receives a randomly generated binary mask indicating which features it can use. Number of features is set by `numAllowedFeatures`.\
(This mask is fixed once per tree, instead of being resampled each split.)

### With the completed set of trees (forest):
- Each tree makes a prediction for a test sample.  
- The final prediction is made with a majority vote from all trees.


## Evaluation
### Parameter choice
How the values were chosen: 
-  `maxRulesPerTree` = 2-5 rules for simple data

-  `numTrees` = $10\log_2(N)$

-  `numAllowedFeatures` = $floor(\sqrt{M})$

Where: 
-  $N$: Number of candidates in the data set.

-  $M$: Number of features per candidate

### Results
The following accuracy results were obtained after training on `training.dat` and testing on `testing.dat`.

With parameter of `maxRulesPerTree` = 5: \
Decision Tree Accuracy:  **63.5%** 

Along with parameters of `numTrees` = 99 and `numAllowedFeatures` = 3: \
Random Forest Accuracy:  **64.1%**

## Future steps
- For random forest, feature subsampling on every node instead of just the tree.
- For random forest, provide an OOB evaluation instead of using training and testing split.
- Modularise the code (e.g. `RandomForest.h/cpp`, `DecisionTree.h/cpp`, `Node.h` etc.) and split header/source.
- Create a synthetic dataset and evaluate against it with a majority class baseline, allowing for recreation and interpretations of results.

