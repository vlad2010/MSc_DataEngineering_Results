using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class Program
{
    static void Main()
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

        // Generate match schedule and save to CSV files
        GenerateMatchSchedule(teams, leagueSetup.Rounds);

        Console.WriteLine("Match schedule generated and saved to CSV files.");
    }

    static void GenerateMatchSchedule(List<Team> teams, int totalRounds)
    {
        int totalTeams = teams.Count;

        if (totalTeams % 2 != 0)
        {
            Console.WriteLine("The number of teams must be even to create a balanced schedule.");
            return;
        }

        if (totalTeams * totalRounds / 2 != 22 * totalTeams)
        {
            Console.WriteLine("The schedule cannot be generated with the specified number of rounds.");
            return;
        }

        for (int roundNumber = 1; roundNumber <= totalRounds; roundNumber++)
        {
            string roundFileName = $"round-{roundNumber:D2}.csv";
            string roundFilePath = Path.Combine("Data", roundFileName);

            // Check if the round file already exists, skip if it does
            if (File.Exists(roundFilePath))
            {
                Console.WriteLine($"Round {roundNumber} already exists. Skipping...");
                continue;
            }

            List<string> matchLines = GenerateRoundMatches(teams, roundNumber);

            // Write the match schedule to the CSV file
            using (StreamWriter writer = new StreamWriter(roundFilePath))
            {
                writer.WriteLine("HomeTeam,AwayTeam,MatchDate,Stadium");

                foreach (string matchLine in matchLines)
                {
                    writer.WriteLine(matchLine);
                }
            }

            Console.WriteLine($"Round {roundNumber} created and saved as {roundFileName}");
        }
    }

    static List<string> GenerateRoundMatches(List<Team> teams, int roundNumber)
    {
        List<string> matchLines = new List<string>();
        int totalTeams = teams.Count;
        int matchesPerRound = totalTeams / 2;

        for (int match = 0; match < matchesPerRound; match++)
        {
            for (int i = 0; i < totalTeams / 2; i++)
            {
                Team homeTeam = teams[i];
                Team awayTeam = teams[totalTeams - 1 - i];

                // Generate match date and stadium (you can customize this part)
                string matchDate = "2023-09-30"; // Modify this with the actual date
                string stadium = $"Stadium {roundNumber}-{match + 1}";

                string matchLine = $"{homeTeam.Abbreviation},{awayTeam.Abbreviation},{matchDate},{stadium}";
                matchLines.Add(matchLine);
            }

            // Rotate teams for the next round
            teams.Insert(1, teams[totalTeams - 1]);
            teams.RemoveAt(totalTeams);
        }

        return matchLines;
    }
}