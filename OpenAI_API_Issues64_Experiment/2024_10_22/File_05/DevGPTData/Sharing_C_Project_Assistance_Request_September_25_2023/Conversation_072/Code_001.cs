using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class Program
{
    static void Main()
    {
        // Load team data from teams.csv
        string teamFilePath = @"C:\Users\kutuk\hey\Football-standings\FootballStandings\Data\teams.csv";
        List<Team> teams = FootballCsvReader.ReadCsv(teamFilePath);

        if (teams == null)
        {
            Console.WriteLine("Unable to load team data. Exiting...");
            return;
        }

        // Specify the number of rounds (in this case, 32)
        int totalRounds = 32;

        // Generate match schedule and save to CSV files
        GenerateMatchSchedule(teams, totalRounds);

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

        // Ensure that each team plays exactly 22 matches
        if (totalTeams * 22 % 2 != 0)
        {
            Console.WriteLine("The number of teams must be divisible by 2 to create a schedule with 22 matches per team.");
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

            List<string> matchLines = GenerateRoundMatches(teams);

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

    static List<string> GenerateRoundMatches(List<Team> teams)
    {
        List<string> matchLines = new List<string>();
        int totalTeams = teams.Count;
        int matchesPerRound = totalTeams / 2;

        // Create a list of team indexes
        List<int> teamIndexes = new List<int>();
        for (int i = 0; i < totalTeams; i++)
        {
            teamIndexes.Add(i);
        }

        // Rotate the team indexes to create matches
        for (int match = 0; match < matchesPerRound; match++)
        {
            List<string> playedAgainst = new List<string>(); // To keep track of teams already played against

            for (int i = 0; i < totalTeams / 2; i++)
            {
                int homeIndex = teamIndexes[i];
                int awayIndex = teamIndexes[totalTeams - 1 - i];

                string homeTeamAbbreviation = teams[homeIndex].Abbreviation;
                string awayTeamAbbreviation = teams[awayIndex].Abbreviation;

                // Ensure that the home team hasn't played against the away team in previous rounds
                if (!playedAgainst.Contains(awayTeamAbbreviation))
                {
                    // Generate match date and stadium (you can customize this part)
                    string matchDate = "2023-09-30"; // Modify this with the actual date
                    string stadium = $"Stadium {match + 1}";

                    string matchLine = $"{homeTeamAbbreviation},{awayTeamAbbreviation},{matchDate},{stadium}";
                    matchLines.Add(matchLine);

                    // Add both teams to the playedAgainst list
                    playedAgainst.Add(homeTeamAbbreviation);
                    playedAgainst.Add(awayTeamAbbreviation);
                }
            }

            // Rotate the team indexes
            teamIndexes.Insert(1, teamIndexes[totalTeams - 1]);
            teamIndexes.RemoveAt(totalTeams);
        }

        return matchLines;
    }
}
