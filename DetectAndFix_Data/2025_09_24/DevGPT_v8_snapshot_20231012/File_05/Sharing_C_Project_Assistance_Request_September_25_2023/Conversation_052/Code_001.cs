case 4:
    // Simulation af alle kampe
    // Test om det virker
    Console.Write("Simulate all matches with random scores? (y/n): ");
    string simulateAllMatches = Console.ReadLine();

    if (simulateAllMatches.ToLower() == "y")
    {
        //"Data" for alle .csv filer
        string dataDirectory = "Data";
        string[] roundFiles = Directory.GetFiles(dataDirectory, "round-*.csv");

        if (roundFiles.Length == 0)
        {
            Console.WriteLine("No round files found in the 'Data' directory.");
        }
        else
        {
            // Sort the round files by their names to ensure processing in order
            Array.Sort(roundFiles);

            foreach (string currentRoundFilePath in roundFiles)
            {
                // Read the CSV file and store the data in a List of strings
                List<string> lines = new List<string>();
                using (StreamReader reader = new StreamReader(currentRoundFilePath))
                {
                    string line;
                    while ((line = reader.ReadLine()) != null)
                    {
                        lines.Add(line);
                    }
                }

                // Generate random scores for each match and update the lines
                Random random = new Random();
                for (int i = 1; i < lines.Count; i++) // Start from index 1 to skip the header row
                {
                    string[] parts = lines[i].Split(',');
                    int homeGoals = random.Next(0, 5); // Generate a random number between 0 and 4 for home team goals
                    int awayGoals = random.Next(0, 5); // Generate a random number between 0 and 4 for away team goals
                    parts[2] = $"{homeGoals}-{awayGoals}"; // Update the score
                    lines[i] = string.Join(",", parts); // Update the line with the new score
                }

                // Write the updated lines back to the CSV file
                using (StreamWriter writer = new StreamWriter(currentRoundFilePath))
                {
                    foreach (string line in lines)
                    {
                        writer.WriteLine(line);
                    }
                }

                Console.WriteLine($"Random scores generated and updated in {Path.GetFileName(currentRoundFilePath)}.");
            }
        }
    }
    else
    {
        Console.WriteLine("Simulation canceled.");
    }
    break;
