private void UpdateStreak(Team homeTeam, Team awayTeam, MatchResult matchResult)
{
    // Determine the match outcome
    bool homeTeamWon = matchResult.HomeTeamGoals > matchResult.AwayTeamGoals;
    bool awayTeamWon = matchResult.AwayTeamGoals > matchResult.HomeTeamGoals;
    bool isDraw = matchResult.HomeTeamGoals == matchResult.AwayTeamGoals;

    // Update home team's streak
    if (homeTeamWon)
    {
        homeTeam.CurrentStreak.Wins++;
        awayTeam.CurrentStreak.Losses = 0; // Reset away team's loss streak
    }
    else if (isDraw)
    {
        homeTeam.CurrentStreak.Draws++;
        awayTeam.CurrentStreak.Draws = 0; // Reset away team's draw streak
        homeTeam.CurrentStreak.Losses = 0; // Reset home team's loss streak
    }
    else
    {
        homeTeam.CurrentStreak.Losses++;
        awayTeam.CurrentStreak.Wins = 0; // Reset away team's win streak
        awayTeam.CurrentStreak.Draws = 0; // Reset away team's draw streak
    }

    // Update away team's streak
    if (awayTeamWon)
    {
        awayTeam.CurrentStreak.Wins++;
        homeTeam.CurrentStreak.Losses = 0; // Reset home team's loss streak
    }
    else if (isDraw)
    {
        awayTeam.CurrentStreak.Draws++;
        homeTeam.CurrentStreak.Draws = 0; // Reset home team's draw streak
        awayTeam.CurrentStreak.Losses = 0; // Reset away team's loss streak
    }
    else
    {
        awayTeam.CurrentStreak.Losses++;
        homeTeam.CurrentStreak.Wins = 0; // Reset home team's win streak
        homeTeam.CurrentStreak.Draws = 0; // Reset home team's draw streak
    }
}
