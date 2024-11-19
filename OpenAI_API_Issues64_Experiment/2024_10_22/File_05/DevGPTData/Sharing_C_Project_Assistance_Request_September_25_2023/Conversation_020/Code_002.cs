private void ResetTeamStatistics()
{
    foreach (var team in teams)
    {
        team.GamesPlayed = 0;
        team.GamesWon = 0;
        team.GamesDrawn = 0;
        team.GamesLost = 0;
        team.GoalsFor = 0;
        team.GoalsAgainst = 0;
        team.Points = 0;
        team.CurrentStreak = new Team.Streak();
    }
}
