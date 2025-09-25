Analysis (security-focused)

- CWE-22 / CWE-73: Path Traversal via user-controlled file name
  - In menu option 2, the program takes roundFileName from Console.ReadLine() and uses Path.Combine("Data", roundFileName). This allows:
    - Traversal with .. or directory separators to escape the Data folder.
    - Absolute paths: on Windows, if the second argument is rooted, Path.Combine ignores the first argument (so a user can force any path).
  - Impact: processing arbitrary files, information disclosure, or DoS.
  - Fix: Strictly validate file names (whitelist pattern like round-<number>.csv), disallow rooted paths, canonicalize and ensure the resolved path is within an expected base directory.

- CWE-426: Untrusted Search Path (reliance on current working directory)
  - Using relative "Data" directory depends on whatever the current working directory is when launching the app. An attacker can influence it to point to an unexpected Data directory with malicious CSVs.
  - Fix: Anchor file access to a trusted base path (e.g., AppContext.BaseDirectory/Data) or a secure configuration path.

- CWE-200: Information Exposure
  - Hard-coded absolute paths expose developer username and directory structure in source control or distributed binaries.
  - Fix: Remove hard-coded absolute paths; derive paths at runtime relative to application base directory or secure configuration.

- CWE-248: Uncaught Exception
  - No try/catch around file I/O or parsing; unexpected errors crash the app (availability issue).
  - Fix: Add targeted exception handling with safe, non-verbose error messages.

- Minor robustness/security hygiene:
  - Use culture-invariant, case-insensitive comparisons (StringComparison.OrdinalIgnoreCase) instead of ToLower() (CWE-178: Improper Handling of Case Sensitivity).
  - When enumerating round files, anchor the directory and optionally filter to a strict filename pattern.

Fixed code (single fragment)

```cs
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

class Program
{
    // Whitelist: round-<number>.csv (e.g., round-1.csv)
    private static readonly Regex RoundFileNameRegex = new Regex(@"^round-\d+\.csv$", RegexOptions.Compiled | RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);

    static void Main(string[] args)
    {
        // Anchor to a trusted base directory to avoid CWE-426 and remove hard-coded absolute paths (CWE-200)
        string baseDataDirectory = Path.Combine(AppContext.BaseDirectory, "Data");

        if (!Directory.Exists(baseDataDirectory))
        {
            Console.WriteLine($"Data directory not found: {baseDataDirectory}");
            return;
        }

        string setupFilePath = Path.Combine(baseDataDirectory, "setup.csv");
        string teamFilePath = Path.Combine(baseDataDirectory, "teams.csv");

        LeagueSetup leagueSetup = null;
        List<Team> teams = null;

        try
        {
            // Load setup data from setup.csv
            if (!File.Exists(setupFilePath))
            {
                Console.WriteLine($"Setup file not found: {setupFilePath}");
                return;
            }
            leagueSetup = LeagueSetup.ParseSetupCsv(setupFilePath);
        }
        catch (Exception)
        {
            Console.WriteLine("Unable to load or parse league setup. Exiting...");
            return;
        }

        if (leagueSetup == null)
        {
            Console.WriteLine("Unable to load league setup. Exiting...");
            return;
        }

        try
        {
            // Load team data from teams.csv
            if (!File.Exists(teamFilePath))
            {
                Console.WriteLine($"Team file not found: {teamFilePath}");
                return;
            }
            teams = FootballCsvReader.ReadCsv(teamFilePath);
        }
        catch (Exception)
        {
            Console.WriteLine("Unable to load or parse team data. Exiting...");
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
                        string? roundFileName = Console.ReadLine();

                        string? roundFilePath = GetSafeRoundFilePath(roundFileName, baseDataDirectory);
                        if (roundFilePath == null)
                        {
                            Console.WriteLine("Invalid round file name. Expected format: round-<number>.csv");
                            break;
                        }

                        try
                        {
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
                        }
                        catch (Exception)
                        {
                            Console.WriteLine("An error occurred while processing the round file.");
                        }
                        break;

                    case 3:
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
                        Console.Write("Simulate all matches? (y/n): ");
                        string? simulateAllMatches = Console.ReadLine();

                        if (string.Equals(simulateAllMatches?.Trim(), "y", StringComparison.OrdinalIgnoreCase))
                        {
                            try
                            {
                                string[] roundFiles = Directory.GetFiles(baseDataDirectory, "round-*.csv", SearchOption.TopDirectoryOnly)
                                    .Where(f => RoundFileNameRegex.IsMatch(Path.GetFileName(f) ?? string.Empty))
                                    .ToArray();

                                if (roundFiles.Length == 0)
                                {
                                    Console.WriteLine("No round files found in the Data directory.");
                                }
                                else
                                {
                                    // Optional: numeric sort for round-<n>.csv to avoid lexicographic misordering
                                    Array.Sort(roundFiles, (a, b) =>
                                    {
                                        int na = ExtractRoundNumber(Path.GetFileName(a));
                                        int nb = ExtractRoundNumber(Path.GetFileName(b));
                                        return na.CompareTo(nb);
                                    });

                                    foreach (string currentRoundFilePath in roundFiles)
                                    {
                                        try
                                        {
                                            processor.GenerateRandomScores(currentRoundFilePath);
                                            processor.ProcessRoundResults(currentRoundFilePath);
                                            Console.WriteLine($"Matches in {Path.GetFileName(currentRoundFilePath)} have been processed.");
                                        }
                                        catch (Exception)
                                        {
                                            Console.WriteLine($"An error occurred while processing {Path.GetFileName(currentRoundFilePath)}. Continuing...");
                                        }
                                    }
                                }
                            }
                            catch (Exception)
                            {
                                Console.WriteLine("An error occurred while enumerating round files.");
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

    // Validate filename and ensure it resolves under the trusted base directory (prevents CWE-22 / CWE-73)
    private static string? GetSafeRoundFilePath(string? userInputFileName, string baseDataDirectory)
    {
        if (string.IsNullOrWhiteSpace(userInputFileName))
        {
            return null;
        }

        string fileName = userInputFileName.Trim();

        // Disallow rooted paths or directory separators
        if (Path.IsPathRooted(fileName) || fileName.IndexOfAny(new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar }) >= 0)
        {
            return null;
        }

        // Enforce whitelist pattern
        if (!RoundFileNameRegex.IsMatch(fileName))
        {
            return null;
        }

        string combined = Path.Combine(baseDataDirectory, fileName);
        string fullBase = EnsureTrailingSeparator(Path.GetFullPath(baseDataDirectory));
        string fullCombined = Path.GetFullPath(combined);

        // Ensure the final path is within the base directory
        if (!fullCombined.StartsWith(fullBase, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return fullCombined;
    }

    private static string EnsureTrailingSeparator(string path)
    {
        if (!path.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal))
        {
            return path + Path.DirectorySeparatorChar;
        }
        return path;
    }

    private static int ExtractRoundNumber(string? fileName)
    {
        if (fileName == null) return int.MaxValue;
        var match = Regex.Match(fileName, @"^round-(\d+)\.csv$", RegexOptions.CultureInvariant | RegexOptions.IgnoreCase);
        if (match.Success && int.TryParse(match.Groups[1].Value, out int n))
        {
            return n;
        }
        return int.MaxValue;
    }
}
```