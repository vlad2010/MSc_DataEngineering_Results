public void DisplayCurrentStandings()
{
    var orderedStandings = teams.OrderByDescending(team => team.Points)
                                .ThenByDescending(team => team.GoalDifference)
                                .ThenByDescending(team => team.GoalsFor)
                                .ToList();

    Console.WriteLine("League Standings:");
    Console.WriteLine("{0,-5} {1,-30} {2,-5} {3,-5} {4,-5} {5,-5} {6,-5} {7,-5} {8,-5} {9,-5} {10,-15}",
                      "Pos", "Team", "Pts", "GP", "W", "D", "L", "GF", "GA", "GD", "Streak");

    for (int i = 0; i < orderedStandings.Count; i++)
    {
        var team = orderedStandings[i];
        string specialMarking = GetSpecialMarking(i + 1);
        string textColor = GetTextColor(i + 1);

        string streakText = $"Wins: {team.CurrentStreak.Wins}, Draws: {team.CurrentStreak.Draws}, Losses: {team.CurrentStreak.Losses}";

        Console.Write("\u001b[0m"); // Reset color
        Console.Write($"\u001b[{textColor}m"); // Set text color

        // Ensure the team name is not cut off
        string teamName = $"{team.Position} {specialMarking} {team.FullName}";
        if (teamName.Length > 30)
        {
            teamName = teamName.Substring(0, 27) + "...";
        }
        else
        {
            teamName = teamName.PadRight(30);
        }

        // Use placeholders to format each column
        Console.WriteLine($"{teamName,-30} {team.Points,-5} {team.GamesPlayed,-5} {team.GamesWon,-5} {team.GamesDrawn,-5} {team.GamesLost,-5} {team.GoalsFor,-5} {team.GoalsAgainst,-5} {team.GoalDifference,-5} {streakText,-15}");

        Console.Write("\u001b[0m"); // Reset color
    }
}
