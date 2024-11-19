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

        List<List<string>> roundMatches = new List<List<string>>();
        List<string> playedAgainst = new List<string>();

        for (int roundNumber = 1; roundNumber <= totalRounds; roundNumber++)
        {
            List<string> matchLines = new List<string>();
            List<int> teamIndexes = new List<int>();
            for (int i = 0; i < totalTeams; i++)
            {
                teamIndexes.Add(i);
            }

            // Rotate the team indexes to create matches
            for (int match = 0; match < totalTeams / 2; match++)
            {
                for (int i = 0; i < totalTeams / 2; i++)
                {
                    int homeIndex = teamIndexes[i];
                    int awayIndex = teamIndexes[totalTeams - 1 - i];

                    string homeTeamAbbreviation = teams[homeIndex].Abbreviation;
                    string awayTeamAbbreviation = teams[awayIndex].Abbreviation;

                    // Ensure that the home team hasn't played against the away team in previous rounds
                    if (!playedAgainst.Contains($"{homeTeamAbbreviation}-{awayTeamAbbreviation}") &&
                        !playedAgainst.Contains($"{awayTeamAbbreviation}-{homeTeamAbbreviation}"))
                    {
                        // Generate match date and stadium (you can customize this part)
                        string matchDate = "2023-09-30"; // Modify this with the actual date
                        string stadium = $"Stadium {roundNumber}";

                        string matchLine = $"{homeTeamAbbreviation},{awayTeamAbbreviation},{matchDate},{stadium}";
                        matchLines.Add(matchLine);

                        // Mark both teams as played against each other
                        playedAgainst.Add($"{homeTeamAbbreviation}-{awayTeamAbbreviation}");
                        playedAgainst.Add($"{awayTeamAbbreviation}-{homeTeamAbbreviation}");
                    }
                }

                // Rotate the team indexes
                teamIndexes.Insert(1, teamIndexes[totalTeams - 1]);
                teamIndexes.RemoveAt(totalTeams);
            }

            roundMatches.Add(matchLines);
        }

        // Write the match schedule to CSV files
        for (int roundNumber = 1; roundNumber <= totalRounds; roundNumber++)
        {
            string roundFileName = $"round-{roundNumber:D2}.csv";
            string roundFilePath = Path.Combine("Data", roundFileName);

            using (StreamWriter writer = new StreamWriter(roundFilePath))
            {
                writer.WriteLine("HomeTeam,AwayTeam,MatchDate,Stadium");

                foreach (string matchLine in roundMatches[roundNumber - 1])
                {
                    writer.WriteLine(matchLine);
                }
            }

            Console.WriteLine($"Round {roundNumber} created and saved as {roundFileName}");
        }
    }
}
