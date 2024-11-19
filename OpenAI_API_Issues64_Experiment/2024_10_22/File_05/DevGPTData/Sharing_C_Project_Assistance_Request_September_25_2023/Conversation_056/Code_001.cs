using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class FootballProcessor
{
    // ... (other class members)

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
}
