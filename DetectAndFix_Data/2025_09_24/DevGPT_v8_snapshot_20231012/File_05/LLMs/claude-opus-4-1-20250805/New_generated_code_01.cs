using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using System.Configuration;

class Program
{
    private static readonly string DataDirectory = GetDataDirectory();
    private static readonly Regex SafeFileNamePattern = new Regex(@"^round-\d+\.csv$", RegexOptions.Compiled);
    private static readonly int MaxFileNameLength = 255;

    static void Main(string[] args)
    {
        try
        {
            // Load setup data from setup.csv using relative or configured path
            string setupFilePath = GetSafeFilePath("setup.csv");
            
            if (!File.Exists(setupFilePath))
            {
                Console.WriteLine("Setup file not found. Please ensure setup.csv exists in the data directory.");
                return;
            }

            LeagueSetup leagueSetup = LeagueSetup.ParseSetupCsv(setupFilePath);

            if (leagueSetup == null)
            {
                Console.WriteLine("Unable to load league setup. Exiting...");
                return;
            }

            // Load team data from teams.csv
            string teamFilePath = GetSafeFilePath("teams.csv");
            
            if (!File.Exists(teamFilePath))
            {
                Console.WriteLine("Teams file not found. Please ensure teams.csv exists in the data directory.");
                return;
            }

            List<Team> teams = FootballCsvReader.ReadCsv(teamFilePath);

            if (teams == null || teams.Count == 0)
            {
                Console.WriteLine("Unable to load team data. Exiting...");
                return;
            }

            // Create an instance of FootballProcessor
            FootballProcessor processor = new FootballProcessor(teams, leagueSetup);

            bool exit = false;
            while (!exit)
            {
                Console.WriteLine("\nFootball Processor Menu:");
                Console.WriteLine("1. Print Teams");
                Console.WriteLine("2. Run Matches");
                Console.WriteLine("3. Display Current League Standings");
                Console.WriteLine("4. Simulate all matches");
                Console.WriteLine("5. Exit");
                Console.Write("Enter your choice: ");

                string input = Console.ReadLine()?.Trim();
                
                if (string.IsNullOrEmpty(input) || !int.TryParse(input, out int choice) || choice < 1 || choice > 5)
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
                        // Implement logic to run matches based on round files
                        Console.Write("Enter the round file name (e.g., round-1.csv): ");
                        string roundFileName = Console.ReadLine()?.Trim();
                        
                        if (string.IsNullOrEmpty(roundFileName))
                        {
                            Console.WriteLine("Invalid file name.");
                            break;
                        }

                        // Validate file name format and prevent path traversal
                        if (!IsValidRoundFileName(roundFileName))
                        {
                            Console.WriteLine("Invalid file name format. Please use format: round-N.csv where N is a number.");
                            break;
                        }

                        string roundFilePath = GetSafeFilePath(roundFileName);

                        if (File.Exists(roundFilePath))
                        {
                            try
                            {
                                // Generate random scores for matches in the round file
                                processor.GenerateRandomScores(roundFilePath);

                                // Process the round results
                                processor.ProcessRoundResults(roundFilePath);

                                Console.WriteLine("Matches have been processed successfully.");
                            }
                            catch (Exception)
                            {
                                Console.WriteLine("An error occurred while processing the matches.");
                            }
                        }
                        else
                        {
                            Console.WriteLine("Round file not found. Please check the file name.");
                        }
                        break;

                    case 3:
                        // Display current league standings
                        try
                        {
                            processor.DisplayCurrentStandings();
                        }
                        catch (Exception)
                        {
                            Console.WriteLine("An error occurred while displaying standings.");
                        }
                        break;

                    case 4:
                        // Simulation of all matches
                        Console.Write("Simulate all matches? (y/n): ");
                        string simulateInput = Console.ReadLine()?.Trim()?.ToLowerInvariant();

                        if (simulateInput == "y")
                        {
                            try
                            {
                                // Get all round files with validation
                                string[] roundFiles = Directory.GetFiles(DataDirectory, "round-*.csv")
                                    .Where(f => IsValidRoundFileName(Path.GetFileName(f)))
                                    .ToArray();

                                if (roundFiles.Length == 0)
                                {
                                    Console.WriteLine("No valid round files found in the data directory.");
                                }
                                else
                                {
                                    // Sort the round files by their names to ensure processing in order
                                    Array.Sort(roundFiles);

                                    foreach (string currentRoundFilePath in roundFiles)
                                    {
                                        try
                                        {
                                            // Generate random scores for matches in the round file
                                            processor.GenerateRandomScores(currentRoundFilePath);

                                            // Process the round results
                                            processor.ProcessRoundResults(currentRoundFilePath);

                                            Console.WriteLine($"Matches in {Path.GetFileName(currentRoundFilePath)} have been processed.");
                                        }
                                        catch (Exception)
                                        {
                                            Console.WriteLine($"Error processing {Path.GetFileName(currentRoundFilePath)}. Skipping...");
                                        }
                                    }
                                }
                            }
                            catch (Exception)
                            {
                                Console.WriteLine("An error occurred while simulating matches.");
                            }
                        }
                        else if (simulateInput != "n")
                        {
                            Console.WriteLine("Invalid input. Please enter 'y' or 'n'.");
                        }
                        else
                        {
                            Console.WriteLine("Simulation canceled.");
                        }
                        break;

                    case 5:
                        // Exit the application
                        exit = true;
                        Console.WriteLine("Exiting application...");
                        break;
                }
            }
        }
        catch (Exception)
        {
            Console.WriteLine("A critical error occurred. The application will now exit.");
            Environment.Exit(1);
        }
    }

    private static string GetDataDirectory()
    {
        // Try to get from app.config first
        string configPath = ConfigurationManager.AppSettings["DataDirectory"];
        if (!string.IsNullOrEmpty(configPath) && Directory.Exists(configPath))
        {
            return Path.GetFullPath(configPath);
        }

        // Fall back to relative path
        string relativePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Data");
        if (!Directory.Exists(relativePath))
        {
            Directory.CreateDirectory(relativePath);
        }
        return relativePath;
    }

    private static string GetSafeFilePath(string fileName)
    {
        // Remove any path components and validate file name
        fileName = Path.GetFileName(fileName);
        
        if (string.IsNullOrEmpty(fileName) || fileName.Length > MaxFileNameLength)
        {
            throw new ArgumentException("Invalid file name");
        }

        // Additional validation for special characters
        if (fileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new ArgumentException("File name contains invalid characters");
        }

        return Path.Combine(DataDirectory, fileName);
    }

    private static bool IsValidRoundFileName(string fileName)
    {
        if (string.IsNullOrEmpty(fileName) || fileName.Length > MaxFileNameLength)
        {
            return false;
        }

        // Check if file name matches expected pattern
        return SafeFileNamePattern.IsMatch(fileName);
    }
}