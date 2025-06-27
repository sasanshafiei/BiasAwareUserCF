#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <queue>
#include <algorithm>
#include <map>

using namespace std;
using namespace chrono;

// For each user-user pair, we store a running sum of residual products and a co-rating count.
struct DotData {
    double sum;  // sum of (residual_u * residual_v)
    int count;   // number of items both users rated
    DotData() : sum(0.0), count(0) {}
};

// We'll store userB, similarity in a max-heap
struct Similarity {
    int userB;
    double value;  // similarity value

    // Flip comparison for max-heap
    bool operator<(const Similarity& other) const {
        return value < other.value;
    }
};

// ------------------- TUNABLE PARAMETERS -------------------
static const int K = 190;           // Keep top-K neighbors
static const double SHRINK = 10.0;  // Significance shrinkage
static const double AMP_FACTOR = 1.3; // Case amplification exponent (was 1.25)
static const int NUM_ITERS = 8;     // Iterations for user/item bias refinement (was 5)

// Gradient-based bias update params
static const double ALPHA = 0.01;   // learning rate
static const double REG   = 0.02;   // regularization
// ----------------------------------------------------------

// Raw cosine similarity
inline double cosineSimilarity(double dotProduct, double magA, double magB) {
    if (magA == 0.0 || magB == 0.0) return 0.0;
    return dotProduct / (sqrt(magA) * sqrt(magB));
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(NULL);

    auto start = high_resolution_clock::now();

    // Training data
    unordered_map<int, unordered_map<int, double>> ratingsByUsers;

    double globalSum = 0.0;
    long long globalCount = 0;
    int maxUserId = 0;
    int maxItemId = 0;

    string line;
    bool isTest = false;

    // 1. Read "train dataset"
    while (true) {
        if (!getline(cin, line)) {
            break; // no more lines
        }
        if (line == "train dataset") {
            isTest = false;
            continue;
        }
        if (line == "test dataset") {
            isTest = true;
            break;
        }

        stringstream ss(line);
        int userId, itemId;
        double rating;
        ss >> userId >> itemId >> rating;

        ratingsByUsers[userId][itemId] = rating;
        globalSum += rating;
        globalCount++;
        if (userId > maxUserId) maxUserId = userId;
        if (itemId > maxItemId) maxItemId = itemId;
    }

    // 2. Compute global mean rating
    double globalMean = (globalCount > 0) ? (globalSum / (double)globalCount) : 3.5;

    // 3. Initialize user and item biases
    // We'll store them, then refine via gradient
    unordered_map<int, double> userBias;
    vector<double> itemBias(maxItemId + 1, 0.0);

    // Initialize biases to 0
    for (auto &uPair : ratingsByUsers) {
        int u = uPair.first;
        userBias[u] = 0.0;
    }

    // 4. Gradient-based bias refinement for NUM_ITERS passes
    // For each pass:
    //   for each user u, for each item i in user u's ratings:
    //       err = rating - (globalMean + userBias[u] + itemBias[i])
    //       userBias[u] += ALPHA * (err - REG * userBias[u])
    //       itemBias[i]  += ALPHA * (err - REG * itemBias[i])
    for (int iter = 0; iter < NUM_ITERS; iter++) {
        for (auto &uPair : ratingsByUsers) {
            int u = uPair.first;
            for (auto &mPair : uPair.second) {
                int i = mPair.first;
                double r = mPair.second;
                double bu = userBias[u];
                double bi = itemBias[i];

                double pred = globalMean + bu + bi;
                double err = r - pred;

                // Gradient update
                userBias[u] += ALPHA * (err - REG * bu);
                itemBias[i]  += ALPHA * (err - REG * bi);
            }
        }
    }

    // 5. Build item->(user, residual) for similarity
    // residual = rating - (globalMean + userBias[u] + itemBias[i])
    vector<vector<pair<int, double>>> itemToUser(maxItemId + 1);
    for (auto &uPair : ratingsByUsers) {
        int u = uPair.first;
        for (auto &mPair : uPair.second) {
            int i = mPair.first;
            double r = mPair.second;
            double baseline = globalMean + userBias[u] + itemBias[i];
            double residual = r - baseline;
            itemToUser[i].push_back({u, residual});
        }
    }

    // 6. Compute dot products
    unordered_map<int, unordered_map<int, DotData>> dotAB;
    unordered_map<int, double> magnitudeMap;

    // Magnitudes
    for (int i = 1; i <= maxItemId; i++) {
        for (auto &p : itemToUser[i]) {
            int u = p.first;
            double res = p.second;
            magnitudeMap[u] += (res * res);
        }
    }

    // Dot products
    for (int i = 1; i <= maxItemId; i++) {
        auto &listU = itemToUser[i];
        for (int a = 0; a < (int)listU.size(); a++) {
            for (int b = a + 1; b < (int)listU.size(); b++) {
                int uA = listU[a].first;
                int uB = listU[b].first;
                double rA = listU[a].second;
                double rB = listU[b].second;
                if (uA > uB) {
                    int tmp = uA;
                    uA = uB;
                    uB = tmp;
                    double tmpR = rA;
                    rA = rB;
                    rB = tmpR;
                }
                dotAB[uA][uB].sum += (rA * rB);
                dotAB[uA][uB].count += 1;
            }
        }
    }

    // 7. Build top-K neighbors with significance weighting + case amplification
    unordered_map<int, priority_queue<Similarity>> topNeighbors;

    for (auto &mapA : dotAB) {
        int uA = mapA.first;
        double magA = magnitudeMap[uA];

        for (auto &mapB : mapA.second) {
            int uB = mapB.first;
            double dotVal = mapB.second.sum;
            int coCount = mapB.second.count;
            double magB = magnitudeMap[uB];

            // raw cosine
            double rawSim = cosineSimilarity(dotVal, magA, magB);
            if (rawSim <= 0.0) continue;

            // significance weighting
            double factor = (double)coCount / ((double)coCount + SHRINK);

            // case amplification
            double ampSim = pow(fabs(rawSim), AMP_FACTOR);
            if (rawSim < 0.0) {
                ampSim = -ampSim; // preserve sign
            }

            double finalSim = ampSim * factor;

            if (finalSim > 0.0) {
                // For uA
                {
                    auto &pqA = topNeighbors[uA];
                    pqA.push({uB, finalSim});
                    if ((int)pqA.size() > K) {
                        pqA.pop();
                    }
                }
                // For uB
                {
                    auto &pqB = topNeighbors[uB];
                    pqB.push({uA, finalSim});
                    if ((int)pqB.size() > K) {
                        pqB.pop();
                    }
                }
            }
        }
    }

    // 8. Process test dataset lines
    while (getline(cin, line)) {
        if (line.empty()) continue;
        stringstream ss(line);
        int userId, itemId;
        ss >> userId >> itemId;

        // Baseline
        double baseline = globalMean;
        if (userBias.find(userId) != userBias.end()) {
            baseline += userBias[userId];
        }
        if (itemId >= 1 && itemId <= maxItemId) {
            baseline += itemBias[itemId];
        }

        double weightedSum = 0.0;
        double sumOfWeights = 0.0;

        auto it = topNeighbors.find(userId);
        if (it != topNeighbors.end()) {
            // copy the PQ
            auto pq = it->second;
            while (!pq.empty()) {
                auto topSim = pq.top();
                pq.pop();
                int neighbor = topSim.userB;
                double sim = topSim.value;

                // if neighbor rated this movie
                auto neighIt = ratingsByUsers[neighbor].find(itemId);
                if (neighIt != ratingsByUsers[neighbor].end()) {
                    double nr = neighIt->second;
                    // neighbor baseline
                    double nb = globalMean + userBias[neighbor];
                    if (itemId >= 1 && itemId <= maxItemId) {
                        nb += itemBias[itemId];
                    }
                    double residual = nr - nb;
                    weightedSum += (residual * sim);
                    sumOfWeights += fabs(sim);
                }
            }
        }

        double prediction = baseline;
        if (sumOfWeights > 0.0) {
            prediction += (weightedSum / sumOfWeights);
        }

        cout << prediction << "\n";
    }

    auto end = high_resolution_clock::now();
    double elapsed = duration_cast<milliseconds>(end - start).count() / 1000.0;
    cerr << "Time elapsed: " << elapsed << " s\n";

    return 0;
}// TIP See CLion help at <a
// href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>.
//  Also, you can try interactive lessons for CLion by selecting
//  'Help | Learn IDE Features' from the main menu.