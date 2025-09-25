#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <limits>

using namespace std;

// Maximum recursion depth to prevent stack overflow
const int MAX_RECURSION_DEPTH = 1000;

/**
 * @brief Secure random number generator
 */
class SecureRandom {
private:
    mt19937 generator;
    uniform_int_distribution<int> distribution;
public:
    SecureRandom(int min, int max) : 
        generator(random_device{}()), 
        distribution(min, max) {}
    
    int generate() {
        return distribution(generator);
    }
};

/**
 * @brief Choose pivot using median-of-three to avoid worst-case scenarios
 */
int choosePivot(vector<int>& arr, int low, int high) {
    if (high - low < 2) {
        return high;
    }
    
    // Prevent integer overflow when calculating middle
    int mid = low + (high - low) / 2;
    
    // Median-of-three: compare first, middle, and last elements
    if (arr[low] > arr[mid]) {
        swap(arr[low], arr[mid]);
    }
    if (arr[low] > arr[high]) {
        swap(arr[low], arr[high]);
    }
    if (arr[mid] > arr[high]) {
        swap(arr[mid], arr[high]);
    }
    
    // Place median at high position for standard partition
    swap(arr[mid], arr[high]);
    return high;
}

/**
 * @brief Secure partition function with pivot selection
 */
int partition(vector<int>& arr, int low, int high) {
    // Validate bounds
    if (low < 0 || high >= static_cast<int>(arr.size()) || low > high) {
        throw out_of_range("Invalid partition bounds");
    }
    
    // Choose pivot intelligently to avoid worst-case scenarios
    int pivotIndex = choosePivot(arr, low, high);
    int pivot = arr[pivotIndex];
    
    int i = low - 1;
    
    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

/**
 * @brief Insertion sort for small subarrays
 */
void insertionSort(vector<int>& arr, int low, int high) {
    for (int i = low + 1; i <= high; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

/**
 * @brief Secure quicksort implementation with depth limiting
 */
void quicksortSecure(vector<int>& arr, int low, int high, int depth) {
    // Prevent stack overflow by limiting recursion depth
    if (depth > MAX_RECURSION_DEPTH) {
        throw runtime_error("Maximum recursion depth exceeded - possible attack or degenerate case");
    }
    
    // Validate bounds
    if (low < 0 || high >= static_cast<int>(arr.size())) {
        throw out_of_range("Invalid sort bounds");
    }
    
    if (low < high) {
        // Use insertion sort for small subarrays (more efficient and secure)
        if (high - low < 10) {
            insertionSort(arr, low, high);
            return;
        }
        
        int pi = partition(arr, low, high);
        
        // Tail call optimization: recurse on smaller partition first
        // This ensures O(log n) stack space in worst case
        if (pi - low < high - pi) {
            quicksortSecure(arr, low, pi - 1, depth + 1);
            quicksortSecure(arr, pi + 1, high, depth + 1);
        } else {
            quicksortSecure(arr, pi + 1, high, depth + 1);
            quicksortSecure(arr, low, pi - 1, depth + 1);
        }
    }
}

/**
 * @brief Public interface for secure quicksort
 */
void quicksort(vector<int>& arr, int low, int high) {
    // Input validation
    if (arr.empty()) {
        return;
    }
    
    if (low < 0 || high >= static_cast<int>(arr.size()) || low > high) {
        throw invalid_argument("Invalid input bounds for quicksort");
    }
    
    // Check for potential integer overflow in array size
    if (arr.size() > static_cast<size_t>(numeric_limits<int>::max() / 2)) {
        throw length_error("Array too large - risk of integer overflow");
    }
    
    try {
        quicksortSecure(arr, low, high, 0);
    } catch (const runtime_error& e) {
        // Fall back to std::sort if recursion depth exceeded
        cerr << "Warning: " << e.what() << " - falling back to std::sort" << endl;
        sort(arr.begin() + low, arr.begin() + high + 1);
    }
}

int main() {
    try {
        const int n = 10; // Size of the array
        
        // Validate array size
        if (n <= 0 || n > 1000000) {
            throw invalid_argument("Invalid array size");
        }
        
        vector<int> arr;
        arr.reserve(n); // Pre-allocate memory for efficiency
        
        // Use cryptographically secure random number generator
        SecureRandom rng(0, 99);
        
        // Fill array with random integers
        for (int i = 0; i < n; i++) {
            arr.push_back(rng.generate());
        }
        
        // Display original array
        cout << "Original array: ";
        for (const auto& val : arr) {
            cout << val << " ";
        }
        cout << endl;
        
        // Sort the array using secure quicksort
        quicksort(arr, 0, n - 1);
        
        // Display sorted array
        cout << "Sorted array: ";
        for (const auto& val : arr) {
            cout << val << " ";
        }
        cout << endl;
        
        // Verify sort correctness
        for (int i = 1; i < n; i++) {
            if (arr[i] < arr[i-1]) {
                throw logic_error("Sort verification failed");
            }
        }
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}