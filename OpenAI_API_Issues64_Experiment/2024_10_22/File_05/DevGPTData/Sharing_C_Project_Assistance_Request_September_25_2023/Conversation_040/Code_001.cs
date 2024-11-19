private void UpdateStreakForTeam(Team team, bool won, bool isDraw)
{
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
}
