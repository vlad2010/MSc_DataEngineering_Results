private void UpdateStreak(Team homeTeam, Team awayTeam, MatchResult matchResult)
{
    // Determine the match outcome
    bool homeTeamWon = matchResult.HomeTeamGoals > matchResult.AwayTeamGoals;
    bool awayTeamWon = matchResult.AwayTeamGoals > matchResult.HomeTeamGoals;
    bool isDraw = matchResult.HomeTeamGoals == matchResult.AwayTeamGoals;

    // Update home team's streak for the current round
    UpdateStreakForRound(homeTeam, homeTeamWon, isDraw);

    // Update away team's streak for the current round
    UpdateStreakForRound(awayTeam, awayTeamWon, isDraw);
}

private void UpdateStreakForRound(Team team, bool won, bool isDraw)
{
    var currentRoundStreak = team.Streaks.LastOrDefault(); // Get the streak for the current round

    if (currentRoundStreak == null)
    {
        // If there is no streak for the current round, create one
        currentRoundStreak = new Team.Streak();
        team.Streaks.Add(currentRoundStreak);
    }

    if (won)
    {
        currentRoundStreak.Wins++;
        team.CurrentStreak.Losses = 0; // Reset loss streak
        team.CurrentStreak.Draws = 0;  // Reset draw streak
    }
    else if (isDraw)
    {
        currentRoundStreak.Draws++;
        team.CurrentStreak.Draws++;
        currentRoundStreak.Wins = 0;   // Reset win streak
        team.CurrentStreak.Wins = 0;   // Reset win streak
        currentRoundStreak.Losses = 0; // Reset loss streak
        team.CurrentStreak.Losses = 0; // Reset loss streak
    }
    else
    {
        currentRoundStreak.Losses++;
        team.CurrentStreak.Wins = 0;   // Reset win streak
        team.CurrentStreak.Draws = 0;  // Reset draw streak
    }
}
