using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class FootballProcessor
{
    private List<Team> originalTeams;
    private LeagueSetup leagueSetup;

    public FootballProcessor(List<Team> teams, LeagueSetup leagueSetup)
    {
        this.originalTeams = teams;
        this.leagueSetup = leagueSetup;
        ResetTeamStatistics();
    }

    public void ProcessRoundResults(string roundFilePath)
    {
        List<MatchResult> matchResults = MatchResultProcessor.ReadMatchResults(roundFilePath);

        foreach (var matchResult in matchResults)
        {
            Team homeTeam = originalTeams.Find(team => team.Abbreviation == matchResult.HomeTeam);
            Team awayTeam = originalTeams.Find(team => team.Abbreviation == matchResult.AwayTeam);

            if (homeTeam != null && awayTeam != null)
            {
                UpdateTeamStatistics(homeTeam, awayTeam, matchResult);
            }
        }

        CalculateStandings();
    }

    public void GenerateRandomScores(string roundFilePath)
    {
        List<MatchResult> matchResults = MatchResultProcessor.ReadMatchResults(roundFilePath);
        Random random = new Random();

        foreach (var matchResult in matchResults)
        {
            // Generate random scores for each match
            int homeTeamGoals = random.Next(0, 5); // Random goals for the home team (0 to 4)
            int awayTeamGoals = random.Next(0, 5); // Random goals for the away team (0 to 4)

            // Update the match result with the random scores
            matchResult.HomeTeamGoals = homeTeamGoals;
            matchResult.AwayTeamGoals = awayTeamGoals;
        }

        // Save the updated match results back to the round file
        MatchResultProcessor.SaveMatchResults(roundFilePath, matchResults);
    }

    // ... (other methods)

    private void ResetTeamStatistics()
    {
        foreach (var team in originalTeams)
        {
            team.GamesPlayed = 0;
            team.GamesWon = 0;
            team.GamesDrawn = 0;
            team.GamesLost = 0;
            team.GoalsFor = 0;
            team.GoalsAgainst = 0;
            team.Points = 0;
            team.CurrentStreak.Wins = 0;
            team.CurrentStreak.Draws = 0;
            team.CurrentStreak.Losses = 0;
        }
    }
}
