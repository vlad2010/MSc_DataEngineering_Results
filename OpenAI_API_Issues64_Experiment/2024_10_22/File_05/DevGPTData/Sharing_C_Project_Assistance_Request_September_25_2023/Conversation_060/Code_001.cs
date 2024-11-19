using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main()
    {
        List<string> teams = new List<string>
        {
            "FCK", "BIF", "FCM", "AaB", "OB", "EfB", "SIF", "RFC", "VB", "AGF", "HOB"
        };

        List<string> matches = GenerateRoundRobinMatches(teams);

        for (int i = 0; i < matches.Count; i++)
        {
            Console.WriteLine($"Round {i + 1}: {matches[i]}");
        }
    }

    static List<string> GenerateRoundRobinMatches(List<string> teams)
    {
        List<string> matches = new List<string>();

        int numTeams = teams.Count;
        int totalRounds = numTeams - 1;
        int matchesPerRound = numTeams / 2;

        List<string> teamList = new List<string>(teams);

        teamList.Remove(teams[0]); // Remove the first team (FCK) to keep it fixed

        for (int round = 0; round < totalRounds; round++)
        {
            List<string> roundMatches = new List<string>();

            int teamIdx = round % (numTeams - 1); // Rotate the teams for each round
            string fixedTeam = teams[0];
            string team1 = teamList[teamIdx];
            string team2 = GetOpponent(teamList, team1, matches);

            roundMatches.Add($"{fixedTeam} vs. {team1}");
            roundMatches.Add($"{team1} vs. {team2}");

            matches.AddRange(roundMatches);
        }

        return matches;
    }

    static string GetOpponent(List<string> teamList, string team1, List<string> matches)
    {
        foreach (var team in teamList)
        {
            string potentialMatch = $"{team1} vs. {team}";
            string reversedMatch = $"{team} vs. {team1}";

            if (!matches.Contains(potentialMatch) && !matches.Contains(reversedMatch))
            {
                teamList.Remove(team);
                return team;
            }
        }

        // Fallback: If no valid opponent is found, return an empty string
        return string.Empty;
    }
}
