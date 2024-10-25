private void UpdateTeamStatistics(Team homeTeam, Team awayTeam, MatchResult matchResult)
{
    // Other updates...

    // Update the current streak for both teams
    UpdateStreak(homeTeam, awayTeam, matchResult);
}
