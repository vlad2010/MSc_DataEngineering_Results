using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

class Program
{
    static void Main()
    {
        List<string> teams = new List<string>
        {
            "FCK", "BIF", "FCM", "AaB", "OB", "EfB", "SIF", "RFC", "VB", "AGF", "HOB"
        };

        List<List<string>> rounds = GenerateRoundRobinSchedule(teams, 22);

        for (int roundNumber = 0; roundNumber < rounds.Count; roundNumber++)
        {
            string roundFileName = $"round-{roundNumber + 1}.csv";
            WriteRoundToCsv(roundFileName, rounds[roundNumber]);
        }
    }

    static List<List<string>> GenerateRoundRobinSchedule(List<string> teams, int matchesPerTeam)
    {
        int numTeams = teams.Count;
        int totalMatches = numTeams * matchesPerTeam;
        int matchesPerRound = numTeams / 2;

        List<List<string>> rounds = new List<List<string>>();

        List<string> teamList = new List<string>(teams);

        for (int roundNumber = 0; roundNumber < totalMatches / matchesPerRound; roundNumber++)
        {
            List<string> roundMatches = new List<string>();

            for (int matchNumber = 0; matchNumber < matchesPerRound; matchNumber++)
            {
                string team1 = teamList[matchNumber];
                string team2 = teamList[numTeams - 1 - matchNumber];

                roundMatches.Add($"{team1} vs. {team2}");
            }

            rounds.Add(roundMatches);

            // Rotate the teams
            string lastTeam = teamList[numTeams - 1];
            teamList.RemoveAt(numTeams - 1);
            teamList.Insert(1, lastTeam);
        }

        return rounds;
    }

    static void WriteRoundToCsv(string fileName, List<string> round)
    {
        using (StreamWriter writer = new StreamWriter(fileName))
        {
            writer.WriteLine("HomeTeam,AwayTeam");

            foreach (string match in round)
            {
                string[] teams = match.Split(new[] { " vs. " }, StringSplitOptions.None);
                writer.WriteLine($"{teams[0]},{teams[1]}");
            }
        }
    }
}
