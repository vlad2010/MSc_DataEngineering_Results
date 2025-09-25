#include <iostream>
#include <vector>
#include <random>
#include <limits>

constexpr int DEFAULT_ARRAY_SIZE = 10;
constexpr int RANDOM_MIN = 0;
constexpr int RANDOM_MAX = 99;

// Partition function with bounds checking
int partition(std::vector<int>& arr, int low, int high) {
    if (low < 0 || high >= static_cast<int>(arr.size()) || low > high) {
        throw std::out_of_range("Invalid indices in partition");
    }
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

// Quicksort function with bounds checking
void quicksort(std::vector<int>& arr, int low, int high) {
    if (low < 0 || high >= static_cast<int>(arr.size()) || low > high) {
        return; // or throw std::out_of_range
    }
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}

int main() {
    int n = DEFAULT_ARRAY_SIZE;

    // If n is user-controlled, validate it
    if (n <= 0 || n > 1000000) { // Arbitrary upper limit for safety
        std::cerr << "Invalid array size." << std::endl;
        return 1;
    }

    std::vector<int> arr;
    arr.reserve(n);

    // Use C++11 random number generation facilities
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(RANDOM_MIN, RANDOM_MAX);

    for (int i = 0; i < n; i++) {
        arr.push_back(distrib(gen));
    }

    quicksort(arr, 0, n - 1);

    for (int i = 0; i < n; i++) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}