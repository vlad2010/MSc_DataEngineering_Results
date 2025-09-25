  #include <iostream>
  #include <fstream>

  int main() {
      // Create an ofstream object and open a file named "example.txt"
      std::ofstream outputFile("example.txt");

      // Check if the file is successfully opened
      if (!outputFile.is_open()) {
          std::cerr << "Error opening file!" << std::endl;
          return 1;
      }

      // Write data to the file
      outputFile << "Hello, File!\n";
      outputFile << "This is a line written to the file.";

      // Close the file
      outputFile.close();

      return 0;
  }