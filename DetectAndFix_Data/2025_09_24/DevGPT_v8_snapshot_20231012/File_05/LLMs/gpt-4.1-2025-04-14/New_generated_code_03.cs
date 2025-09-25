using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;

class Program
{
    static void Main(string[] args)
    {
        // Use relative paths or configurable paths
        string dataDirectory = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data");
        string setupFilePath = Path.Combine(dataDirectory, "setup.csv");
        string teamFilePath = Path.Combine(dataDirectory, "teams.csv");

        LeagueSetup leagueSetup = null;
        List<Team> teams = null;

        try
        {
            leagueSetup = LeagueSetup.ParseSetupCsv(setupFilePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Unable to load league setup. Exiting...");
            // Optionally log ex.Message securely
            return;
        }

        if (leagueSetup == null)
        {
            Console.WriteLine("Unable to load league setup. Exiting...");
            return;
        }

        try
        {
            teams = FootballCsvReader.ReadCsv(teamFilePath);
        }
        catch (Exception ex)
        {
            Console.WriteLine("Unable to load team data. Exiting...");
            // Optionally log ex.Message securely
            return;
        }

        if (teams == null)
        {
            Console.WriteLine("Unable to load team data. Exiting...");
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
            Console.Write("Enter your choice: ");

            if (int.TryParse(Console.ReadLine(), out int choice))
            {
                switch (choice)
                {
                    case 1:
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

                        // Validate file name: only allow round-<number>.csv
                        if (!Regex.IsMatch(roundFileName, @"^round-\d+\.csv$", RegexOptions.IgnoreCase))
                        {
                            Console.WriteLine("Invalid file name format. Please use 'round-<number>.csv'.");
                            break;
                        }

                        string roundFilePath = Path.Combine(dataDirectory, roundFileName);

                        // Ensure the file is within the Data directory
                        if (!IsFileInDirectory(roundFilePath, dataDirectory))
                        {
                            Console.WriteLine("Invalid file path.");
                            break;
                        }

                        if (File.Exists(roundFilePath))
                        {
                            try
                            {
                                processor.GenerateRandomScores(roundFilePath);
                                processor.ProcessRoundResults(roundFilePath);
                                Console.WriteLine("Matches have been processed.");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("Error processing matches.");
                                // Optionally log ex.Message securely
                            }
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
                            try
                            {
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
                                        // Ensure the file is within the Data directory
                                        if (!IsFileInDirectory(currentRoundFilePath, dataDirectory))
                                        {
                                            Console.WriteLine($"Skipping invalid file: {Path.GetFileName(currentRoundFilePath)}");
                                            continue;
                                        }

                                        processor.GenerateRandomScores(currentRoundFilePath);
                                        processor.ProcessRoundResults(currentRoundFilePath);

                                        Console.WriteLine($"Matches in {Path.GetFileName(currentRoundFilePath)} have been processed.");
                                    }
                                }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("Error during simulation.");
                                // Optionally log ex.Message securely
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
            else
            {
                Console.WriteLine("Invalid input. Please enter a valid menu choice.");
            }
        }
    }

    // Helper to ensure file is within the intended directory
    private static bool IsFileInDirectory(string filePath, string directoryPath)
    {
        var fullFilePath = Path.GetFullPath(filePath);
        var fullDirectoryPath = Path.GetFullPath(directoryPath);

        return fullFilePath.StartsWith(fullDirectoryPath, StringComparison.OrdinalIgnoreCase);
    }
}