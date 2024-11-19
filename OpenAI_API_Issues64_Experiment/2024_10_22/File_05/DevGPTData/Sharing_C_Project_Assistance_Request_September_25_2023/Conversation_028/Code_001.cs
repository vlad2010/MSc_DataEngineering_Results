public class Team
{
    // Other properties...

    public List<MatchResult> MatchHistory { get; set; } // Store the match history for streak calculation

    public Team()
    {
        // Initialize the match history list
        MatchHistory = new List<MatchResult>();
    }
}

public class FootballProcessor
{
    // Other fields...

    public void ProcessRoundResults(string roundFilePath)
    {
        ResetTeamStatistics();
        List<MatchResult> matchResults = MatchResultProcessor.ReadMatchResults(roundFilePath);

        foreach (var matchResult in matchResults)
        {
            Team homeTeam = originalTeams.Find(team => team.Abbreviation == matchResult.HomeTeam);
            Team awayTeam = originalTeams.Find(team => team.Abbreviation == matchResult.AwayTeam);

            if (homeTeam != null && awayTeam != null)
            {
                UpdateTeamStatistics(homeTeam, awayTeam, matchResult);
                UpdateStreak(homeTeam, awayTeam, matchResult);
            }
        }

        CalculateStandings();
    }

    private void UpdateStreak(Team homeTeam, Team awayTeam, MatchResult matchResult)
    {
        // Determine the match outcome
        bool homeTeamWon = matchResult.HomeTeamGoals > matchResult.AwayTeamGoals;
        bool awayTeamWon = matchResult.AwayTeamGoals > matchResult.HomeTeamGoals;
        bool isDraw = matchResult.HomeTeamGoals == matchResult.AwayTeamGoals;

        // Update home team's streak
        homeTeam.MatchHistory.Add(matchResult);

        if (homeTeam.MatchHistory.Count > 1)
        {
            MatchResult previousMatch = homeTeam.MatchHistory[homeTeam.MatchHistory.Count - 2];

            if (homeTeamWon && previousMatch.HomeTeamGoals > previousMatch.AwayTeamGoals)
            {
                homeTeam.CurrentStreak.Wins++;
            }
            else if (isDraw && previousMatch.HomeTeamGoals == previousMatch.AwayTeamGoals)
            {
                homeTeam.CurrentStreak.Draws++;
            }
            else if (!homeTeamWon && !isDraw)
            {
                homeTeam.CurrentStreak.Losses++;
            }
        }

        // Update away team's streak
        awayTeam.MatchHistory.Add(matchResult);

        if (awayTeam.MatchHistory.Count > 1)
        {
            MatchResult previousMatch = awayTeam.MatchHistory[awayTeam.MatchHistory.Count - 2];

            if (awayTeamWon && previousMatch.AwayTeamGoals > previousMatch.HomeTeamGoals)
            {
                awayTeam.CurrentStreak.Wins++;
            }
            else if (isDraw && previousMatch.AwayTeamGoals == previousMatch.HomeTeamGoals)
            {
                awayTeam.CurrentStreak.Draws++;
            }
            else if (!awayTeamWon && !isDraw)
            {
                awayTeam.CurrentStreak.Losses++;
            }
        }
    }
}
