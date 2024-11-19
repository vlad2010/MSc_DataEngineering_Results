using System;
using System.Collections.Generic;
using System.IO;
using System.Security;
using System.Text.RegularExpressions;

class Program
{
    static void Main(string[] args)
    {
        // Load paths from configuration or environment variables
        string setupFilePath = Environment.GetEnvironmentVariable("SETUP_FILE_PATH") ?? @"setup.csv";
        LeagueSetup leagueSetup;
        try
        {
            leagueSetup = LeagueSetup.ParseSetupCsv(setupFilePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading setup data: {ex.Message}");
            return;
        }

        string teamFilePath = Environment.GetEnvironmentVariable("TEAM_FILE_PATH") ?? @"teams.csv";
        List<Team> teams;
        try
        {
            teams = FootballCsvReader.ReadCsv(teamFilePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading team data: {ex.Message}");
            return;
        }

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
            Console.Write("Enter your choice (1-5): ");

            string userInput = Console.ReadLine();
            if (!int.TryParse(userInput, out int choice) || choice < 1 || choice > 5)
            {
                Console.WriteLine("Invalid input. Please enter a number between 1 and 5.");
                continue;
            }

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
                    Console.Write("Enter the round file name (e.g., round-1.csv): ");
                    string roundFileName = Console.ReadLine();
                    string roundFilePath = Path.Combine("Data", SanitizeFileName(roundFileName));

                    if (File.Exists(roundFilePath))
                    {
                        processor.GenerateRandomScores(roundFilePath);
                        processor.ProcessRoundResults(roundFilePath);
                        Console.WriteLine("Matches have been processed.");
                    }
                    else
                    {
                        Console.WriteLine("Round file not found. Please check the file name.");
                    }
                    break;

                case 3:
                    processor.DisplayCurrentStandings();
                    break;

                case 4:
                    Console.Write("Simulate all matches? (y/n): ");
                    string simulateAllMatches = Console.ReadLine();

                    if (simulateAllMatches.ToLower() == "y")
                    {
                        string dataDirectory = "Data";
                        string[] roundFiles = Directory.GetFiles(dataDirectory, "round-*.csv");

                        if (roundFiles.Length == 0)
                        {
                            Console.WriteLine("No round files found in the 'Data' directory.");
                        }
                        else
                        {
                            Array.Sort(roundFiles);

                            foreach (string currentRoundFilePath in roundFiles)
                            {
                                processor.GenerateRandomScores(currentRoundFilePath);
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
                    exit = true;
                    break;

                default:
                    Console.WriteLine("Invalid choice. Please select a valid option.");
                    break;
            }
        }
    }

    private static string SanitizeFileName(string fileName)
    {
        return Regex.Replace(fileName, @"[^a-zA-Z0-9\-\.]", string.Empty);
    }
}