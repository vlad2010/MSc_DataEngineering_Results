    int myArray[5];
    for (size_t i = 0; i < sizeof(myArray) / sizeof(myArray[0]); ++i) {
        // Access elements using size_t as the index
        std::cout << myArray[i] << std::endl;
    }