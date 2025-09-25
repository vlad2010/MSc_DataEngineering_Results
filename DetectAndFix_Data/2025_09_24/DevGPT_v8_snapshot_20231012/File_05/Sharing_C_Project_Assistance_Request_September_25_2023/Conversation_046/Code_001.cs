private void UpdateStreak(Team homeTeam, Team awayTeam, MatchResult matchResult)
{
    bool homeTeamWon = matchResult.HomeTeamGoals > matchResult.AwayTeamGoals;
    bool awayTeamWon = matchResult.AwayTeamGoals > matchResult.HomeTeamGoals;
    bool isDraw = matchResult.HomeTeamGoals == matchResult.AwayTeamGoals;

    Console.WriteLine($"Match: {homeTeam.FullName} vs. {awayTeam.FullName}");
    Console.WriteLine($"Home Team Won: {homeTeamWon}");
    Console.WriteLine($"Away Team Won: {awayTeamWon}");
    Console.WriteLine($"Is Draw: {isDraw}");

    UpdateStreakForTeam(homeTeam, homeTeamWon, isDraw);
    UpdateStreakForTeam(awayTeam, awayTeamWon, isDraw);
}

private void UpdateStreakForTeam(Team team, bool won, bool isDraw)
{
    Console.WriteLine($"Updating streak for {team.FullName}");
    
    if (won)
    {
        team.CurrentStreak.Wins++;
        team.CurrentStreak.Draws = 0;
        team.CurrentStreak.Losses = 0;
    }
    else if (isDraw)
    {
        team.CurrentStreak.Draws++;
        team.CurrentStreak.Wins = 0;
        team.CurrentStreak.Losses = 0;
    }
    else
    {
        team.CurrentStreak.Losses++;
        team.CurrentStreak.Wins = 0;
        team.CurrentStreak.Draws = 0;
    }

    Console.WriteLine($"Streak for {team.FullName}: Wins={team.CurrentStreak.Wins}, Draws={team.CurrentStreak.Draws}, Losses={team.CurrentStreak.Losses}");
}
