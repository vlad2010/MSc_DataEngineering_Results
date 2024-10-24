using System;
using System.Collections.Generic;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        // Load setup data from setup.csv
        string setupFilePath = @"C:\Users\kutuk\hey\Football-standings\FootballStandings\Data\setup.csv";
        LeagueSetup leagueSetup = LeagueSetup.ParseSetupCsv(setupFilePath);

        if (leagueSetup == null)
        {
            Console.WriteLine("Unable to load league setup. Exiting...");
            return;
        }

        // Load team data from teams.csv
        string teamFilePath = @"C:\Users\kutuk\hey\Football-standings\FootballStandings\Data\teams.csv";
        List<Team> teams = FootballCsvReader.ReadCsv(teamFilePath);

        if (teams == null)
        {
            Console.WriteLine("Unable to load team data. Exiting...");
            return;
        }

        // Create an instance of FootballProcessor
        FootballProcessor processor = new FootballProcessor(teams, leagueSetup);

        bool exit = false;
        while (!exit)
        {
            Console.WriteLine("Football Processor Menu:");
            Console.WriteLine("1. Print Teams");
            Console.WriteLine("2. Run Matches");
            Console.WriteLine("3. Display Current League Standings");
            Console.WriteLine("4. Simulate all matches");
            Console.WriteLine("5. Exit");
            Console.Write("Enter your choice: ");

            if (int.TryParse(Console.ReadLine(), out int choice))
            {
                switch (choice)
                {
                    case 1:
                        // Print teams and their details
                        foreach (var team in teams)
                        {
                            Console.WriteLine($"Abbreviation: {team.Abbreviation}");
                            // ... (print other team details)
                            Console.WriteLine();
                        }
                        break;

                    case 2:
                        // Implement logic to run matches based on round files
                        Console.Write("Enter the round file name (e.g., round-1.csv): ");
                        string roundFileName = Console.ReadLine();
                        string roundFilePath = Path.Combine("Data", roundFileName);

                        if (File.Exists(roundFilePath))
                        {
                            // Generate random scores for matches in the round file
                            processor.GenerateRandomScores(roundFilePath);

                            // Process the round results
                            processor.ProcessRoundResults(roundFilePath);

                            Console.WriteLine("Matches have been processed.");
                        }
                        else
                        {
                            Console.WriteLine("Round file not found. Please check the file name.");
                        }
                        break;

                    case 3:
                        // Display current league standings
                        processor.DisplayCurrentStandings();
                        break;

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
                                // Sort the round files by their names to ensure processing in order
                                Array.Sort(roundFiles);

                                foreach (string currentRoundFilePath in roundFiles)
                                {
                                    // Generate random scores for matches in the round file
                                    processor.GenerateRandomScores(currentRoundFilePath);

                                    // Process the round results
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

                    case 5:
                        // Exit the application
                        exit = true;
                        break;

                    default:
                        Console.WriteLine("Invalid choice. Please select a valid option.");
                        break;
                }
            }
            else
            {
                Console.WriteLine("Invalid input. Please enter a valid menu choice.");
            }
        }
    }
}
