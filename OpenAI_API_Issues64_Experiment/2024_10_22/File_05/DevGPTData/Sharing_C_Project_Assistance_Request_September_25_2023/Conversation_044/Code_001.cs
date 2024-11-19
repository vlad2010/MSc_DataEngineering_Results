case 4:
    // Simulation af alle kampe
    // Test om det virker
    Console.Write("Simulate all matches? (y/n): ");
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
            // Create a new FootballProcessor instance before the loop
            FootballProcessor processor = new FootballProcessor(teams, leagueSetup);

            // Sort the round files by their names to ensure processing in order
            Array.Sort(roundFiles);

            foreach (string currentRoundFilePath in roundFiles)
            {
                // Use the existing FootballProcessor instance inside the loop
                processor.ProcessRoundResults(currentRoundFilePath);
                Console.WriteLine($"Matches in {Path.GetFileName(currentRoundFilePath)} have been processed.");
            }
        }
    }
    else
    {
        Console.WriteLine("Simulation canceled.");
    }
    break;
