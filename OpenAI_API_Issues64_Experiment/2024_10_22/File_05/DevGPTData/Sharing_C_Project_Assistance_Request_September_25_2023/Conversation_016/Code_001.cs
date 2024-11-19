public void DisplayCurrentStandings()
{
    var orderedStandings = teams.OrderByDescending(team => team.Points)
                                .ThenByDescending(team => team.GoalDifference)
                                .ThenByDescending(team => team.GoalsFor)
                                .ToList();

    Console.WriteLine("League Standings:");
    Console.WriteLine("{0,-5} {1,-35} {2,-5} {3,-5} {4,-5} {5,-5} {6,-5} {7,-5} {8,-5} {9,-5} {10,-15}",
                      "Pos", "Team", "Pts", "GP", "W", "D", "L", "GF", "GA", "GD", "Streak");

    for (int i = 0; i < orderedStandings.Count; i++)
    {
        var team = orderedStandings[i];
        string specialMarking = GetSpecialMarking(i + 1);

        string teamName = $"{team.Position} {specialMarking} {team.FullName}";
        if (teamName.Length > 35)
        {
            teamName = teamName.Substring(0, 32) + "...";
        }

        string streakText = $"Wins: {team.CurrentStreak.Wins}, Draws: {team.CurrentStreak.Draws}, Losses: {team.CurrentStreak.Losses}";

        string textColor = GetTextColor(i + 1);
        Console.Write("\u001b[0m"); // Reset color
        Console.Write($"\u001b[{textColor}m"); // Set text color

        Console.WriteLine($"{team.Position,-5} {teamName,-35} {team.Points,-5} {team.GamesPlayed,-5} {team.GamesWon,-5} {team.GamesDrawn,-5} {team.GamesLost,-5} {team.GoalsFor,-5} {team.GoalsAgainst,-5} {team.GoalDifference,-5} {streakText,-15}");

        Console.Write("\u001b[0m"); // Reset color
    }
}

private string GetTextColor(int position)
{
    if (position == 1)
        return "32"; // Green for first place
    if (position <= leagueSetup.PromoteToChampionsLeague)
        return "36"; // Cyan for qualification places
    if (position >= teams.Count - leagueSetup.RelegateToLowerLeague)
        return "31"; // Red for relegation-threatened teams
    return "0"; // Default color
}