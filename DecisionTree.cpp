#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <queue>
#include <random>
#include <functional>
#include <algorithm>

using namespace std;

constexpr int NUM_FEATURES = 10; 
constexpr int NUM_CLASSES = 3;

// Node Classes
// Abstract base class
class Node {
public:
    virtual ~Node() {}
    
    // Predict using features
    virtual int predict(const vector<int>& features) const = 0;
    
    // Print the node
    virtual void print(int depth = 0) const = 0;
};

// Leaf node stores the indices belonging to it 
// and a predicted class from majority vote
class LeafNode : public Node {
public:
    vector<int> dataIndices;    // indices into the training data
    int predictedLabel;         // majority label which is found after tree building

    LeafNode(const vector<int>& indices)
        : dataIndices(indices), predictedLabel(-1) {}

    virtual int predict(const vector<int>& /*features*/) const override {
        return predictedLabel;
    }

    virtual void print(int depth = 0) const override {
        for (int i = 0; i < depth; i++) cout << "  ";
        cout << "Leaf (label=" << predictedLabel << ") [";
        
        for (size_t i = 0; i < dataIndices.size(); i++) {
            cout << dataIndices[i];
            if (i != dataIndices.size() - 1)
                cout << ",";
        }
        cout << "]\n";
    }
};

// Rule node holds a single feature rule (feature can be 0 or 1 only, -1 for missing)
// and pointers to left and right child/subtrees
class RuleNode : public Node {
public:
    int feature;            // feature index used for splitting
    bool missingGoesLeft;   // decision for missing data at this node: if true, missing values go left
    Node* left;             // branch for feature value 0 (or missing if missingGoesLeft is true)
    Node* right;            // branch for feature value 1 (or missing if missingGoesLeft is false)

    RuleNode(int feat, bool missingLeft)
        : feature(feat), missingGoesLeft(missingLeft), left(nullptr), right(nullptr) {}

    virtual int predict(const vector<int>& features) const override {
        int val = features[feature];
        // recursive until LeafNode, which actually returns label
        if (val == 0) {
            return left->predict(features);
        } else if (val == 1) {
            return right->predict(features);
        } else { // treat -1 or anything invalid as missing
            return missingGoesLeft ? left->predict(features) : right->predict(features);
        }
    }

    ~RuleNode() override { delete left; delete right; }

    virtual void print(int depth = 0) const override {
        for (int i = 0; i < depth; i++) cout << "  ";
        cout << "Rule: Feature[" << feature << "]? (missing goes " << (missingGoesLeft ? "left[0]" : "right[1]") << ")\n";
        
        for (int i = 0; i < depth; i++) cout << "  ";
        cout << "-> [0]:\n";
        if (left) left->print(depth + 1);
        
        for (int i = 0; i < depth; i++) cout << "  ";
        cout << "-> [1]:\n";
        if (right) right->print(depth + 1);
    }
};


// Impurity (gini) and helper functions
// Return Gini from a list of class labels (0, 1, 2)
double gini(const vector<int>& labels) {
    
    int count[NUM_CLASSES] = {};
    for (int label : labels)
        count[label]++;
    
    int total = labels.size();
    if (total == 0) return 0.0;
    
    double g = 0.0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (count[i] > 0) {
            double p = double(count[i]) / total;
            g += p * p;
        }
    }
    
    return 1 - g;
}

// Given indices, extract their class labels from the data
// Data row have 10 features, then a class label
vector<int> extractLabels(const vector<vector<int>>& data, const vector<int>& indices) {
    vector<int> labels;
    for (int idx : indices) {
        labels.push_back(data[idx].back());
    }
    return labels;
}

// Return majority label from a vector of labels
int majorityLabel(const vector<int>& labels) {
    int freq[NUM_CLASSES] = {};
    for (int label : labels)
        freq[label]++;
    
    int maj = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (freq[i] > freq[maj])
            maj = i;
    }
    
    return maj;
}


// Candidate Split Computation
// Structure to hold a candidate split (how a leaft ptr would be split)
struct SplitCandidate {
    Node** leafPtr;             // Pointer to the pointer, pointing the current leaf node
    int splitFeature;           // the features to split on
    bool missingGoesLeft;       // missing values go left if true else false
    double improvement;         // = (weighted Gini after split) - (current Gini)
    vector<int> leftIndices;    // Data indices goes to left branch
    vector<int> rightIndices;   // Data indices goes to right branch

    SplitCandidate(Node** ptr, int feat, bool missingLeft, double imp,
                   const vector<int>& leftI, const vector<int>& rightI)
        : leafPtr(ptr), splitFeature(feat), missingGoesLeft(missingLeft),
          improvement(imp), leftIndices(leftI), rightIndices(rightI) {}
};

// helper comparator for a min–heap to choose candidate with most negative (best) improvement
struct CandidateComparator {
    bool operator()(const SplitCandidate& a, const SplitCandidate& b) {
        return a.improvement > b.improvement;
    }
};

// finds best candidate split for the given leaf node
// returns true if a valid split is found
bool bestSplit(const vector<vector<int>>& data, LeafNode* leaf, Node** leafPtr,
                const vector<bool>& allowedMask, SplitCandidate& bestCandidate) {
    
    // allowedMask is which features the tree is allowed to use
    const vector<int>& indices = leaf->dataIndices;
    if (indices.empty()) return false;

    vector<int> currLabels = extractLabels(data, indices);
    double currentImpurity = gini(currLabels);
    // nothing to gain from splitting a pure leaf
    if (currentImpurity == 0.0) return false;  

    bool foundCandidate = false;
    double bestImp = numeric_limits<double>::infinity();
    int bestFeat = -1;
    bool bestMissingLeft = true;
    vector<int> bestLeftIndices, bestRightIndices;

    // For each allowed feature
    // And for each missingToLeft (0 or 1)
    // sort all items by feat=0 or feat=1 (and -1 handling)
    for (int f = 0; f < NUM_FEATURES; f++) {
        if (!allowedMask[f])
            continue;
        
        for (int missingToLeft = 0; missingToLeft < 2; missingToLeft++) {
            bool missingLeft = (missingToLeft == 0);
            vector<int> leftIndices, rightIndices;
            
            for (int idx : indices) {
                int feat = data[idx][f];
                if (feat == 0) {
                    leftIndices.push_back(idx);
                } else if (feat == 1) {
                    rightIndices.push_back(idx);
                } else { // treat -1 or others as missing or invalid
                    (missingLeft ? leftIndices : rightIndices).push_back(idx);
                }
            }
            
            // only consider valid splits, i.e. when both branches are nonempty
            if (leftIndices.empty() || rightIndices.empty())
                continue;
            
            vector<int> leftLabels = extractLabels(data, leftIndices);
            vector<int> rightLabels = extractLabels(data, rightIndices);
            double leftGini = gini(leftLabels);
            double rightGini = gini(rightLabels);
            double total = indices.size();
            double weightedGini = (leftIndices.size() / total) * leftGini +
                                  (rightIndices.size() / total) * rightGini;
            double improvement = weightedGini- currentImpurity;
            
            if (improvement < bestImp) { // more negative is better
                bestImp = improvement;
                bestFeat = f;
                bestMissingLeft = missingLeft;
                bestLeftIndices = leftIndices;
                bestRightIndices = rightIndices;
                foundCandidate = true;
            }
        }
    }
    if (foundCandidate) {
        bestCandidate = SplitCandidate(leafPtr, bestFeat, bestMissingLeft, bestImp, 
                                        bestLeftIndices, bestRightIndices);
    }
    
    return foundCandidate;
}


// Tree building for a single tree (best first), returns root node
// rootIndices = which rows to start the tree with
// allowedMask = which features the tree is allowed to use (for random forest)
// maxRuleNodes = maximum number of rule nodes/splits per tree
Node* buildDecisionTree(const vector<vector<int>>& data, const vector<int>& rootIndices,
                        const vector<bool>& allowedMask, int maxRuleNodes) {
    // start with a root leaf with specified indices
    Node* root = new LeafNode(rootIndices);

    // priority queue for candidate splits (top = most negative (best improvement))
    priority_queue<SplitCandidate, vector<SplitCandidate>, CandidateComparator> candidateQueue;

    // for root node
    Node** rootPtr = &root;
    LeafNode* rootLeaf = dynamic_cast<LeafNode*>(*rootPtr);
    if (rootLeaf) {
        SplitCandidate candidate(nullptr, -1, true, 0, vector<int>(), vector<int>());
        if (bestSplit(data, rootLeaf, rootPtr, allowedMask, candidate))
            candidateQueue.push(candidate);
    }
    
    // for rest of children nodes
    int ruleCount = 0;
    while (ruleCount < maxRuleNodes && !candidateQueue.empty()) {
        SplitCandidate candidate = candidateQueue.top();
        candidateQueue.pop();

        LeafNode* candidateLeaf = dynamic_cast<LeafNode*>(*(candidate.leafPtr));
        if (!candidateLeaf)
            continue; // candidate outdated

        // create new RuleNode based on candidate
        RuleNode* rule = new RuleNode(candidate.splitFeature, candidate.missingGoesLeft);
        rule->left = new LeafNode(candidate.leftIndices);
        rule->right = new LeafNode(candidate.rightIndices);
        *(candidate.leafPtr) = rule;
        delete candidateLeaf;
        ruleCount++;

        // find best candidate for:
        // left child
        Node** leftPtr = &(rule->left);
        LeafNode* leftLeaf = dynamic_cast<LeafNode*>(*leftPtr);
        if (leftLeaf) {
            SplitCandidate leftCandidate(nullptr, -1, true, 0, vector<int>(), vector<int>());
            
            if (bestSplit(data, leftLeaf, leftPtr, allowedMask, leftCandidate))
                candidateQueue.push(leftCandidate);
        }
        // right child
        Node** rightPtr = &(rule->right);
        LeafNode* rightLeaf = dynamic_cast<LeafNode*>(*rightPtr);
        if (rightLeaf) {
            SplitCandidate rightCandidate(nullptr, -1, true, 0, vector<int>(), vector<int>());
            
            if (bestSplit(data, rightLeaf, rightPtr, allowedMask, rightCandidate))
                candidateQueue.push(rightCandidate);
        }
    }

    // after building the tree, assign predicted labels only to each leaf by majority vote
    function<void(Node*)> assignLabels = [&](Node* node) {
        RuleNode* rule = dynamic_cast<RuleNode*>(node);
        if (rule) {
            assignLabels(rule->left);
            assignLabels(rule->right);
        } else {
            LeafNode* leaf = dynamic_cast<LeafNode*>(node);
            if (leaf) {
                vector<int> labels = extractLabels(data, leaf->dataIndices);
                leaf->predictedLabel = majorityLabel(labels);
            }
        }
    };
    assignLabels(root);
    return root;
}


// random forest
class RandomForest {
public:
    vector<Node*> trees;
    int numTrees;
    int maxRulesPerTree;
    vector<vector<bool>> allowedMasks; // one mask per tree

    // build forest using the training data
    // numAllowedFeatures = how many features each tree can use
    RandomForest(const vector<vector<int>>& trainData, int numTrees_, int maxRulesPerTree_, int numAllowedFeatures)
        : numTrees(numTrees_), maxRulesPerTree(maxRulesPerTree_)
    {
        int totalFeatures = NUM_FEATURES;
        random_device rd;
        mt19937 gen(rd());  // seed random number generator with random seed
        
        for (int i = 0; i < numTrees; i++) {
            
            // bootstrapping
            uniform_int_distribution<size_t> rowDist(0, trainData.size() - 1);
            vector<int> bootstrapIndices(trainData.size());
            for (size_t j = 0; j < trainData.size(); j++)
                bootstrapIndices[j] = rowDist(gen);


            // shuffle the ascending indices list, 
            // grab the first `numAllowedFeatures` items for indices 
            // for the feature mask to mark true
            vector<bool> mask(totalFeatures, false);
            vector<int> indices(totalFeatures);
            
            for (int j = 0; j < totalFeatures; j++)
                indices[j] = j;
            shuffle(indices.begin(), indices.end(), gen);
            
            for (int j = 0; j < numAllowedFeatures; j++)
                mask[indices[j]] = true;
            allowedMasks.push_back(mask);
            
            // build a tree using allowed mask
            Node* tree = buildDecisionTree(trainData, bootstrapIndices, mask, maxRulesPerTree);
            trees.push_back(tree);
        }
    }

    // predict class label for one test data point
    int predict(const vector<int>& features) const {
        int votes[NUM_CLASSES] = {};
        
        // for each tree in forest, get majority vote of leaf nodes
        for (Node* tree : trees) {
            int pred = tree->predict(features);
            votes[pred]++;
        }
        
        int maj = 0;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (votes[i] > votes[maj])
                maj = i;
        }
        return maj;
    }

    // destructor free all allocated trees
    ~RandomForest() {
        for (Node* tree : trees)
            delete tree;
    }

    // print all trees in forest
    void printForest() const {
        for (size_t i = 0; i < trees.size(); i++) {
            cout << "Tree " << i << ":\n";
            trees[i]->print();
            cout << "\n";
        }
    }
};


// Data Reading
// Each line must have 11 integers (10 features and 1 class label)
bool readData(const string& filename, vector<vector<int>>& data) {
    ifstream infile(filename);
    
    if (!infile.is_open()) {
        cout << "Error: Could not open file " << filename << "\n";
        return false;
    }
    
    string line;
    while(getline(infile, line)) {
        if (line.empty())
            continue;
        
        istringstream iss(line);
        vector<int> point;
        int value;
        
        while(iss >> value)
            point.push_back(value);
        
        // also catches any irregularities causing iss >> value to break early
        if (point.size() != NUM_FEATURES + 1) {
            cout << "Warning: A line does not have " << NUM_FEATURES + 1 << "items; this line is skipped\n";
            continue;
        }
        
        int label = point.back();
        if (label < 0 || label >= NUM_CLASSES){
            cout << "Warning: invalid class label " << label << "; this line is skipped\n";
            continue;
        }

        for (int i=0; i < NUM_FEATURES; i++){
            if (point[i] < -1 || point[i] > 1){
                cout << "Warning: invalid feature value " << point[i] << ", is not in {-1, 0, 1}\n";
                break;  // accepted since it is handled in the code
            }
        }

        data.push_back(point);
    }
    
    infile.close();

    if (data.size() == 0){
        cout << "Error: File " << filename << " is empty\n";
        return false;
    }
    
    return true;
}

// Evaluatio
// Evaluate accuracy for a tree (or forest) on test data
double evalModel(const Node* model, const vector<vector<int>>& testData) {
    int correct = 0;
    
    for (const auto& point : testData) {
        vector<int> features(point.begin(), point.begin() + NUM_FEATURES);
        int trueLabel = point.back();
        int pred = model->predict(features);
        if (pred == trueLabel)
            correct++;
    }
    
    return double(correct) / testData.size();
}

// Overloaded function for RandomForest
double evalForest(const RandomForest& forest, const vector<vector<int>>& testData) {
    int correct = 0;
    
    for (const auto& point : testData) {
        vector<int> features(point.begin(), point.begin() + NUM_FEATURES);
        int trueLabel = point.back();
        int pred = forest.predict(features);
        if (pred == trueLabel)
            correct++;
    }
    
    return double(correct) / testData.size();
}

// Main Build and Evaluate Models
int main() {
    
    vector<vector<int>> trainData;
    vector<vector<int>> testData;
    if (!readData("training.dat", trainData))
        return 1;
    if (!readData("testing.dat", testData))
        return 1;


    // ====================
    // single decision tree
    // ====================
    
    // use all features for the decision tree (allowed mask: all true).
    vector<bool> allFeatures(NUM_FEATURES, true);
    
    // can adjust the tree depth around 2-15 (number of rule nodes)
    int maxRulesPerTree = 5;

    vector<int> allIndices(trainData.size());
    for (size_t i = 0; i < trainData.size(); i++)
        allIndices[i] = i;

    Node* decisionTree = buildDecisionTree(trainData, allIndices, allFeatures, maxRulesPerTree);

    // uncomment to print single decision tree 
    // cout << "Single Decision Tree Structure:\n";
    // decisionTree->print();

    double treeAccuracy = evalModel(decisionTree, testData);
    cout << "Decision Tree Accuracy: " << treeAccuracy * 100 << "%\n";

    delete decisionTree;


    // ====================
    // random forest
    // ====================

    int numTrees = 99;              // 10 * log2(N), where N is the number of training samples
    int numAllowedFeatures = 3;     // floor(sqrt(M)), where M is features
    RandomForest forest(trainData, numTrees, maxRulesPerTree, numAllowedFeatures);

    // uncomment to print forest trees: 
    // forest.printForest();

    double forestAccuracy = evalForest(forest, testData);
    cout << "Random Forest Accuracy: " << forestAccuracy * 100 << "%\n";

    // forest destroyed automatically on return
    return 0;
}
